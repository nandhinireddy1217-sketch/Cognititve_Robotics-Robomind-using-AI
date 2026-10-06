
import os
import time
import tempfile
import threading

import av
import cv2
import ollama
import plotly.graph_objects as go
import streamlit as st

from backend.voice import listen_to_voice, speak_text
from backend.analytics import generate_sensor_data
from rag.knowledge_base import search_knowledge
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RoboMind",
    page_icon="🤖",
    layout="wide",
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "messages": [],
    "battery": 85,
    "command_history": [],
    "robot_status": "STOPPED",
    "sensor_history": [],
    "robot_action": "Idle",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def execute_command(command):
    """Execute a simulated robot movement command."""
    st.session_state.command_history.append(command)
    st.session_state.robot_status = command
    st.session_state.robot_action = command

    if command != "STOP" and st.session_state.battery > 5:
        st.session_state.battery -= 1


def parse_detection(result):
    """
    Read the exact detection fields requested from Gemma's report.
    Returns:
        person_detected, device_detected, object_detected,
        warning_detected, warning_level
    """
    text = result.lower()

    person_detected = "person: yes" in text
    device_detected = "device: yes" in text
    object_detected = "object: yes" in text
    warning_detected = "warning: yes" in text

    warning_level = "NONE"
    for level in ("HIGH", "MEDIUM", "LOW", "NONE"):
        if f"warning level:\n{level.lower()}" in text:
            warning_level = level
            break
        if f"warning level: {level.lower()}" in text:
            warning_level = level
            break

    # A MEDIUM/HIGH level always counts as a warning.
    if warning_level in ("MEDIUM", "HIGH"):
        warning_detected = True

    return (
        person_detected,
        device_detected,
        object_detected,
        warning_detected,
        warning_level,
    )


def build_warning_message(
    person_detected,
    device_detected,
    object_detected,
    warning_detected,
    warning_level,
):
    """Create a useful alert without treating every object as dangerous."""

    detections = []

    if person_detected:
        detections.append("👤 Person detected")

    if device_detected:
        detections.append("📱 Device detected")

    if object_detected:
        detections.append("📦 Object detected")

    detection_text = " | ".join(detections)

    if warning_detected:
        if detection_text:
            return f"🚨 SAFETY WARNING — {warning_level} | {detection_text}"
        return f"🚨 SAFETY WARNING — {warning_level}"

    # Detection information is not a safety warning.
    if detection_text:
        return f"👁️ Detected: {detection_text}"

    return ""


# =========================================================
# CONTINUOUS CAMERA + AI MONITOR
# =========================================================

class CameraProcessor(VideoProcessorBase):

    def __init__(self):
        self.latest_frame = None
        self.last_analysis = "Waiting for AI analysis..."
        self.warning_message = ""
        self.warning_active = False
        self.warning_level = "NONE"

        self.last_analysis_time = 0.0
        self.analysis_running = False

        self.person_detected = False
        self.device_detected = False
        self.object_detected = False

        self.lock = threading.Lock()

    def recv(self, frame):

        # -----------------------------------------------------
        # Convert camera frame
        # -----------------------------------------------------

        img = frame.to_ndarray(format="bgr24")

        with self.lock:
            self.latest_frame = img.copy()

        # -----------------------------------------------------
        # Analyze approximately every 4 seconds
        # -----------------------------------------------------

        current_time = time.time()

        with self.lock:
            should_analyze = (
                current_time - self.last_analysis_time >= 4
                and not self.analysis_running
            )

            if should_analyze:
                self.last_analysis_time = current_time
                self.analysis_running = True

        if should_analyze:
            frame_copy = img.copy()

            threading.Thread(
                target=self.analyze_frame,
                args=(frame_copy,),
                daemon=True,
            ).start()

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24",
        )

    def analyze_frame(self, frame):

        image_path = None

        try:
            # -------------------------------------------------
            # CREATE TEMPORARY IMAGE
            # -------------------------------------------------

            temp_file = tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False,
            )

            image_path = temp_file.name
            temp_file.close()

            success = cv2.imwrite(
                image_path,
                frame,
            )

            if not success:
                raise RuntimeError("Could not save camera frame.")

            # -------------------------------------------------
            # GEMMA 3 VISION
            # -------------------------------------------------

            prompt = """
You are RoboMind's real-time cognitive vision system.

Analyze the camera image carefully and provide a COMPLETE, DETAILED
description of everything that can actually be seen.

Do not give a short or one-line answer.
Do not guess names, identities, intentions, locations, or facts that cannot
be confirmed from the image.

IMPORTANT:
- A normal visible person, device, or object is NOT automatically dangerous.
- Only mark WARNING: YES when there is a visible safety issue, unusual
  situation, or clearly concerning condition.
- If there is no obvious safety issue, use WARNING: NO and
  WARNING LEVEL: NONE.

First provide these detection fields exactly:

PERSON: YES or NO
DEVICE: YES or NO
OBJECT: YES or NO
WARNING: YES or NO

Then provide a detailed report using exactly these sections:

OVERALL SCENE:
Describe the complete visible scene, setting, lighting, and layout.

PEOPLE DETECTED:
Describe the number of visible people and their visible appearance,
position, pose, and activity. Do not identify a person by name.

OBJECTS DETECTED:
List and describe important visible objects and where they are located.

DEVICES DETECTED:
List and describe visible electronic or other devices and their positions.

ACTIVITY:
Describe what the visible people or objects appear to be doing, only when
it can reasonably be determined from the image.

ENVIRONMENT:
Describe the visible surroundings, background, room or outdoor setting,
and other relevant environmental details.

IMPORTANT DETAILS:
Mention notable visual details that RoboMind should remember from this frame.

SAFETY CONCERNS:
Describe any visible safety issue or unusual situation.
If there is none, say:
"No obvious safety concern visible."

WARNING LEVEL:
Choose exactly one:
NONE, LOW, MEDIUM, HIGH.

FINAL SUMMARY:
Give a complete but concise summary of the entire image.

Make the report detailed enough that a user who cannot see the camera image
can understand what RoboMind observed.
"""

            response = ollama.chat(
                model="gemma3:4b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_path],
                    }
                ],
            )

            result = response["message"]["content"]

            # -------------------------------------------------
            # PARSE DETECTIONS
            # -------------------------------------------------

            (
                person_detected,
                device_detected,
                object_detected,
                warning_detected,
                warning_level,
            ) = parse_detection(result)

            warning = build_warning_message(
                person_detected,
                device_detected,
                object_detected,
                warning_detected,
                warning_level,
            )

            # -------------------------------------------------
            # SAVE LATEST FRAME
            # -------------------------------------------------

            cv2.imwrite(
                "captured_image.jpg",
                frame,
            )

            # -------------------------------------------------
            # UPDATE PROCESSOR STATE
            # -------------------------------------------------

            with self.lock:
                self.last_analysis = result
                self.warning_message = warning
                self.warning_active = warning_detected
                self.warning_level = warning_level

                self.person_detected = person_detected
                self.device_detected = device_detected
                self.object_detected = object_detected

        except Exception as e:

            with self.lock:
                self.last_analysis = f"Vision AI error: {e}"
                self.warning_message = "⚠️ Vision system error"
                self.warning_active = True
                self.warning_level = "HIGH"

        finally:

            with self.lock:
                self.analysis_running = False

            if image_path:
                try:
                    os.remove(image_path)
                except OSError:
                    pass


# =========================================================
# TITLE
# =========================================================

st.title("🤖 RoboMind")

st.subheader("Intelligent Cognitive Robot")

st.write(
    "AI-powered robot with RAG, Llama 3.2, Gemma 3 Vision and ChromaDB"
)


# =========================================================
# ROBO MIND DASHBOARD
# =========================================================

st.divider()

st.header("🏠 RoboMind Dashboard")

st.write(
    "Monitor the AI, robot, voice, vision and sensor systems."
)


# =========================================================
# DASHBOARD MAIN STATUS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🧠 AI Model",
        "Llama 3.2",
        "Online",
    )

with col2:
    st.metric(
        "👁️ Vision AI",
        "Gemma 3:4b",
        "Ready",
    )

with col3:
    st.metric(
        "📚 Knowledge Base",
        "ChromaDB",
        "Active",
    )

with col4:
    st.metric(
        "🔋 Robot Battery",
        f"{st.session_state.battery}%",
        "Simulation",
    )

st.progress(
    max(0.0, min(1.0, st.session_state.battery / 100))
)


# =========================================================
# ROBOT STATUS
# =========================================================

if st.session_state.battery > 50:
    st.success("🟢 Robot Status: Online")
elif st.session_state.battery > 20:
    st.warning("🟡 Robot Status: Low Battery")
else:
    st.error("🔴 Robot Status: Critical Battery")


# =========================================================
# LAST ROBOT COMMAND
# =========================================================

if st.session_state.command_history:

    last_command = st.session_state.command_history[-1]

    st.info(
        f"🎮 Last Robot Command: {last_command}"
    )

else:

    st.info(
        "🎮 Last Robot Command: No command yet"
    )


# =========================================================
# QUICK ROBOT CONTROLS
# =========================================================

st.subheader("🎮 Quick Robot Controls")

control1, control2, control3, control4, control5 = st.columns(5)

with control1:
    if st.button(
        "⬆️ Forward",
        width="stretch",
        key="quick_forward",
    ):
        execute_command("FORWARD")
        st.success("🤖 Moving FORWARD")

with control2:
    if st.button(
        "⬅️ Left",
        width="stretch",
        key="quick_left",
    ):
        execute_command("LEFT")
        st.success("🤖 Moving LEFT")

with control3:
    if st.button(
        "⏹️ Stop",
        width="stretch",
        key="quick_stop",
    ):
        execute_command("STOP")
        st.warning("🛑 Robot STOPPED")

with control4:
    if st.button(
        "➡️ Right",
        width="stretch",
        key="quick_right",
    ):
        execute_command("RIGHT")
        st.success("🤖 Moving RIGHT")

with control5:
    if st.button(
        "⬇️ Backward",
        width="stretch",
        key="quick_backward",
    ):
        execute_command("BACKWARD")
        st.success("🤖 Moving BACKWARD")


# =========================================================
# SENSOR SUMMARY
# =========================================================

st.subheader("📡 Current Sensor Status")

try:
    dashboard_sensor = generate_sensor_data()
except Exception as e:
    dashboard_sensor = {
        "temperature": 0,
        "distance": 0,
        "motion": 0,
        "battery": st.session_state.battery,
    }
    st.warning(f"Sensor system error: {e}")

sensor1, sensor2, sensor3, sensor4 = st.columns(4)

with sensor1:
    st.metric(
        "🌡️ Temperature",
        f"{dashboard_sensor.get('temperature', 0)} °C",
    )

with sensor2:
    st.metric(
        "📏 Distance",
        f"{dashboard_sensor.get('distance', 0)} cm",
    )

with sensor3:
    motion_text = (
        "Detected"
        if safe_int(dashboard_sensor.get("motion", 0)) == 1
        else "No Motion"
    )

    st.metric(
        "🚶 Motion",
        motion_text,
    )

with sensor4:
    st.metric(
        "🔋 Sensor Battery",
        f"{dashboard_sensor.get('battery', st.session_state.battery)}%",
    )


# =========================================================
# AI CHATBOT
# =========================================================

st.divider()

st.header("💬 RoboMind AI Chatbot")


# =========================================================
# DISPLAY PREVIOUS CHAT
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask RoboMind something..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.write(question)

    # -----------------------------------------------------
    # RAG SEARCH
    # -----------------------------------------------------

    with st.spinner("🔍 Searching knowledge base..."):

        try:
            documents = search_knowledge(
                question,
                n_results=3,
            )

            if documents:
                context = "\n\n".join(
                    str(document)
                    for document in documents
                )
            else:
                context = "No relevant information found."

        except Exception as e:
            st.error(f"Knowledge base error: {e}")
            context = ""

    # -----------------------------------------------------
    # RAG PROMPT
    # -----------------------------------------------------

    prompt = f"""
You are RoboMind, an intelligent cognitive robot.

Use the following knowledge base information to answer the user's question.

KNOWLEDGE BASE:
{context}

USER QUESTION:
{question}

Instructions:
- Answer clearly.
- Use simple language.
- Use the knowledge base when relevant.
- Do not invent information.
- If the information is not available, clearly say that it is not
  available in the knowledge base.
"""

    # -----------------------------------------------------
    # ASK LLAMA
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🤖 RoboMind is thinking..."):

            try:
                response = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )

                answer = response["message"]["content"]

                st.write(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as e:
                st.error(f"AI error: {e}")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🤖 RoboMind")

    st.write("Intelligent Cognitive Robot")

    st.divider()

    st.subheader("🧠 AI Components")

    st.write("✅ Llama 3.2")
    st.write("✅ Ollama")
    st.write("✅ ChromaDB")
    st.write("✅ RAG")
    st.write("✅ Gemma 3 Vision")
    st.write("✅ Voice Assistant")
    st.write("✅ Always-On Camera")

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        width="stretch",
    ):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# VOICE ASSISTANT + RAG
# =========================================================

st.divider()

st.header("🎤 RoboMind Voice Assistant")

st.write(
    "Speak to RoboMind and get an AI response using the RoboMind "
    "knowledge base."
)


if st.button(
    "🎙️ Speak to RoboMind",
    width="stretch",
):

    # -----------------------------------------------------
    # LISTEN
    # -----------------------------------------------------

    with st.spinner("🎤 Listening..."):

        try:
            voice_question = listen_to_voice()
        except Exception as e:
            voice_question = ""
            st.error(f"Voice input error: {e}")

    if voice_question:

        st.write("🗣️ You said:")
        st.info(voice_question)

        if not voice_question.lower().startswith("sorry"):

            # -------------------------------------------------
            # SEARCH KNOWLEDGE BASE
            # -------------------------------------------------

            with st.spinner(
                "📚 Searching RoboMind knowledge base..."
            ):

                try:
                    documents = search_knowledge(
                        voice_question,
                        n_results=3,
                    )

                    if documents:
                        context = "\n\n".join(
                            str(document)
                            for document in documents
                        )
                    else:
                        context = "No relevant information found."

                except Exception as e:
                    st.error(f"Knowledge base error: {e}")
                    context = ""

            # -------------------------------------------------
            # CREATE PROMPT
            # -------------------------------------------------

            prompt = f"""
You are RoboMind, an intelligent cognitive robot.

Use the following knowledge base to answer the user's question.

KNOWLEDGE BASE:
{context}

USER QUESTION:
{voice_question}

Instructions:
- Answer clearly.
- Use simple language.
- Use the knowledge base when relevant.
- Do not invent information.
- If the answer is not available in the knowledge base,
  say that the information is not available in the knowledge base.
"""

            # -------------------------------------------------
            # ASK LLAMA
            # -------------------------------------------------

            with st.spinner("🧠 RoboMind is thinking..."):

                try:
                    response = ollama.chat(
                        model="llama3.2",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                    )

                    answer = response["message"]["content"]

                    st.subheader("🤖 RoboMind:")
                    st.write(answer)

                    # -------------------------------------------------
                    # SPEAK ANSWER
                    # -------------------------------------------------

                    with st.spinner("🔊 Speaking..."):
                        try:
                            speak_text(answer)
                        except Exception as e:
                            st.warning(
                                f"Text-to-speech error: {e}"
                            )

                except Exception as e:
                    st.error(f"AI error: {e}")

    else:
        st.warning("No voice question was received.")


# =========================================================
# ALWAYS-ON VISION CAMERA
# =========================================================

st.divider()

st.header("👁️ RoboMind Vision AI")

st.write(
    "The camera stays ON continuously. Gemma 3:4b automatically analyzes "
    "the live camera approximately every 4 seconds, detects people, "
    "devices and objects, and generates a complete detailed vision report."
)


# =========================================================
# LIVE CAMERA
# =========================================================

st.subheader("📹 Live Camera")

webrtc_ctx = webrtc_streamer(
    key="robot-camera",
    video_processor_factory=CameraProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
    desired_playing_state=True,
)


# =========================================================
# CONTINUOUS VISION MONITOR
# =========================================================

def display_vision_monitor():

    processor = webrtc_ctx.video_processor

    if processor is None:
        st.info(
            "📷 Waiting for camera... Please allow browser camera access "
            "when the browser asks for permission."
        )
        return

    with processor.lock:

        latest_frame = (
            processor.latest_frame.copy()
            if processor.latest_frame is not None
            else None
        )

        latest_result = processor.last_analysis
        warning = processor.warning_message
        warning_active = processor.warning_active
        warning_level = processor.warning_level
        analysis_running = processor.analysis_running

        person_detected = processor.person_detected
        device_detected = processor.device_detected
        object_detected = processor.object_detected

    # -----------------------------------------------------
    # WARNING / DETECTION STATUS
    # -----------------------------------------------------

    if warning_active:
        st.error(warning or "🚨 Safety warning detected.")
    else:
        detection_items = []

        if person_detected:
            detection_items.append("👤 Person")

        if device_detected:
            detection_items.append("📱 Device")

        if object_detected:
            detection_items.append("📦 Object")

        if detection_items:
            st.info(
                "👁️ Detected: "
                + " | ".join(detection_items)
                + " — No safety warning."
            )
        else:
            st.success(
                "🟢 No person, device or object detection warning."
            )

    status1, status2, status3 = st.columns(3)

    with status1:

        if latest_frame is not None:
            st.success("🟢 Camera: ON")
        else:
            st.warning("🟡 Camera: Waiting")

    with status2:

        if analysis_running:
            st.info("🧠 Gemma 3:4b: Analyzing...")
        else:
            st.success("🧠 Gemma 3:4b: Monitoring")

    with status3:

        if warning_level in ("MEDIUM", "HIGH"):
            st.error(f"⚠️ Warning Level: {warning_level}")
        elif warning_level == "LOW":
            st.warning("⚠️ Warning Level: LOW")
        else:
            st.success("✅ Warning Level: NONE")

    # -----------------------------------------------------
    # LATEST FRAME
    # -----------------------------------------------------

    if latest_frame is not None:

        rgb_frame = cv2.cvtColor(
            latest_frame,
            cv2.COLOR_BGR2RGB,
        )

        st.image(
            rgb_frame,
            caption="🔴 Latest RoboMind Camera Frame",
            width="stretch",
        )

    # -----------------------------------------------------
    # COMPLETE VISION REPORT
    # -----------------------------------------------------

    st.subheader("🤖 Complete Vision AI Analysis")

    if latest_result:
        st.markdown(latest_result)
    else:
        st.info(
            "Waiting for the first detailed camera analysis..."
        )

    if latest_frame is not None:
        st.caption(
            "💾 Latest analyzed frame is automatically saved as "
            "`captured_image.jpg`."
        )


# ---------------------------------------------------------
# Streamlit fragment keeps the monitor UI refreshed.
# ---------------------------------------------------------

if hasattr(st, "fragment"):

    @st.fragment(run_every="2s")
    def vision_monitor_fragment():
        display_vision_monitor()

    vision_monitor_fragment()

else:
    display_vision_monitor()


# =========================================================
# ROBOT SIMULATION
# =========================================================

st.divider()

st.header("🤖 RoboMind Robot Control")

st.write(
    "Control the robot using Simulation Mode."
)


# =========================================================
# ROBOT STATUS
# =========================================================

st.success(
    "🟢 Robot Status: Simulation Mode"
)

st.metric(
    "🔋 Battery",
    f"{st.session_state.battery}%",
)

st.info(
    f"🎮 Current Movement: {st.session_state.robot_status}"
)


# =========================================================
# ROBOT MOVEMENT CONTROL
# =========================================================

st.subheader("🎮 Robot Movement Control")

col1, col2, col3 = st.columns(3)

with col1:

    if st.button(
        "⬅️ Left",
        width="stretch",
        key="movement_left",
    ):
        execute_command("LEFT")
        st.session_state.robot_action = "Moving Left"

with col2:

    if st.button(
        "⬆️ Forward",
        width="stretch",
        key="movement_forward",
    ):
        execute_command("FORWARD")
        st.session_state.robot_action = "Moving Forward"

with col3:

    if st.button(
        "➡️ Right",
        width="stretch",
        key="movement_right",
    ):
        execute_command("RIGHT")
        st.session_state.robot_action = "Moving Right"

col4, col5, col6 = st.columns(3)

with col5:

    if st.button(
        "⬇️ Backward",
        width="stretch",
        key="movement_backward",
    ):
        execute_command("BACKWARD")
        st.session_state.robot_action = "Moving Backward"

st.write(
    f"🤖 Current Action: **{st.session_state.robot_action}**"
)


# =========================================================
# BATTERY HEALTH
# =========================================================

st.subheader("🔋 Battery Health")

battery = st.session_state.battery

if battery > 50:

    st.success(
        f"🟢 Battery Healthy — {battery}%"
    )

elif battery > 20:

    st.warning(
        f"🟡 Battery Warning — {battery}%"
    )

else:

    st.error(
        f"🔴 Battery Critical — {battery}%"
    )

st.progress(
    max(0.0, min(1.0, battery / 100))
)


# =========================================================
# SYSTEM HEALTH
# =========================================================

st.subheader("🩺 System Health")

health1, health2, health3, health4 = st.columns(4)

with health1:
    st.success(
        "🟢 Ollama\n\nConfigured"
    )

with health2:
    st.success(
        "🟢 RAG\n\nActive"
    )

with health3:
    st.success(
        "🟢 Vision\n\nReady"
    )

with health4:
    st.success(
        "🟢 Sensors\n\nOnline"
    )


# =========================================================
# AI SYSTEM ACTIVITY
# =========================================================

st.subheader("🧠 AI System Activity")

ai1, ai2, ai3, ai4, ai5 = st.columns(5)

with ai1:
    st.success(
        "🧠 Llama 3.2\n\nChatbot"
    )

with ai2:
    st.success(
        "👁️ Gemma 3:4b\n\nVision"
    )

with ai3:
    st.success(
        "📚 ChromaDB\n\nRAG"
    )

with ai4:
    st.success(
        "🎤 Voice\n\nActive"
    )

with ai5:
    st.success(
        "📊 Analytics\n\nLive"
    )


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

st.header("📋 RoboMind Project Information")

info1, info2, info3 = st.columns(3)

with info1:

    st.info(
        "🤖 **Project Type**\n\n"
        "AI + Full Stack + Robotics"
    )

with info2:

    st.info(
        "🧠 **AI Technology**\n\n"
        "Ollama + Llama 3.2 + Gemma 3"
    )

with info3:

    st.info(
        "📚 **Knowledge System**\n\n"
        "RAG + ChromaDB"
    )


# =========================================================
# ROBOT MOVEMENT
# =========================================================

st.subheader("🎮 Robot Movement")


if st.button(
    "⬆️ Forward",
    width="stretch",
    key="sim_forward",
):

    execute_command("FORWARD")

    st.success(
        "🤖 Robot moving FORWARD"
    )


col1, col2, col3 = st.columns(3)

with col1:

    if st.button(
        "⬅️ Left",
        width="stretch",
        key="sim_left",
    ):

        execute_command("LEFT")

        st.success(
            "🤖 Robot moving LEFT"
        )

with col2:

    if st.button(
        "⏹️ Stop",
        width="stretch",
        key="sim_stop",
    ):

        execute_command("STOP")

        st.warning(
            "🛑 Robot STOPPED"
        )

with col3:

    if st.button(
        "➡️ Right",
        width="stretch",
        key="sim_right",
    ):

        execute_command("RIGHT")

        st.success(
            "🤖 Robot moving RIGHT"
        )


if st.button(
    "⬇️ Backward",
    width="stretch",
    key="sim_backward",
):

    execute_command("BACKWARD")

    st.success(
        "🤖 Robot moving BACKWARD"
    )


# =========================================================
# COMMAND HISTORY
# =========================================================

st.subheader("📜 Command History")

if st.session_state.command_history:

    for command in reversed(
        st.session_state.command_history[-10:]
    ):

        st.write(
            f"➡️ {command}"
        )

else:

    st.info(
        "No commands yet."
    )


# =========================================================
# ROBOT ACTIVITY ANALYTICS
# =========================================================

st.subheader("📊 Robot Activity Analytics")

command_counts = {}

for command in st.session_state.command_history:

    command_counts[command] = (
        command_counts.get(command, 0) + 1
    )


if command_counts:

    activity_chart = go.Figure()

    activity_chart.add_trace(
        go.Bar(
            x=list(command_counts.keys()),
            y=list(command_counts.values()),
            name="Commands",
        )
    )

    activity_chart.update_layout(
        xaxis_title="Robot Command",
        yaxis_title="Number of Times Used",
        height=350,
    )

    st.plotly_chart(
        activity_chart,
        width="stretch",
    )

else:

    st.info(
        "No robot activity recorded yet."
    )


# =========================================================
# RECENT ROBOT ACTIVITY
# =========================================================

st.subheader("🕒 Recent Robot Activity")

if st.session_state.command_history:

    recent_commands = (
        st.session_state.command_history[-5:]
    )

    for i, command in enumerate(
        reversed(recent_commands),
        1,
    ):

        st.write(
            f"**{i}.** 🎮 Robot Command: `{command}`"
        )

else:

    st.info(
        "No robot activity recorded yet."
    )


# =========================================================
# ROBOT PERFORMANCE SUMMARY
# =========================================================

st.subheader("📈 Robot Performance Summary")

total_commands = len(
    st.session_state.command_history
)


if st.session_state.command_history:

    command_counts = {}

    for command in st.session_state.command_history:

        command_counts[command] = (
            command_counts.get(command, 0) + 1
        )

    most_used_command = max(
        command_counts,
        key=command_counts.get,
    )

else:

    most_used_command = "None"


summary1, summary2, summary3 = st.columns(3)

with summary1:

    st.metric(
        "🎮 Total Commands",
        total_commands,
    )

with summary2:

    st.metric(
        "🚀 Most Used Command",
        most_used_command,
    )

with summary3:

    st.metric(
        "🔋 Current Battery",
        f"{st.session_state.battery}%",
    )


# =========================================================
# RESET SIMULATION
# =========================================================

if st.button(
    "🔄 Reset Simulation",
    width="stretch",
):

    st.session_state.battery = 85
    st.session_state.command_history = []
    st.session_state.robot_status = "STOPPED"
    st.session_state.robot_action = "Idle"

    st.success(
        "✅ Simulation reset!"
    )


# =========================================================
# REAL-TIME ANALYTICS
# =========================================================

st.divider()

st.header("📊 RoboMind Real-Time Analytics")

st.write(
    "Monitor robot sensor data and visualize it using live graphs."
)
