import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import time
import supervision as sv
import os
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from PyPDF2 import PdfReader

st.set_page_config(layout="wide")

load_dotenv()

groq_api_key = os.environ.get('GROQ_API_KEY')

yolo_model = None
cap = None
filtered_detections = [] # Global variable to store detected objects

def initialize_yolo_and_camera():
    global yolo_model, cap
    yolo_model = YOLO("best.pt")
    frame_width, frame_height = 640, 360  # Change as needed
    cap = cv2.VideoCapture(0) #0 is built-in or first camera and 1 is plug and play
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


def detect_objects(frame):
    global filtered_detections  # Use global keyword
    result = yolo_model(frame)[0]
    detections = sv.Detections.from_yolov8(result)

    filtered_detections = []
    for detection in detections:
        _, confidence, class_id, _ = detection
        if confidence > 0.7:
            class_name = yolo_model.model.names[class_id]
            if class_name in ['Arduino', 'Breadboard']:
                filtered_detections.append(detection)

    print("Filtered Detections during object detection:", filtered_detections)
    labels = [
        f"{yolo_model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _
        in filtered_detections
    ]

    for detection, label in zip(filtered_detections, labels):
        x, y, w, h = map(int, detection[0])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, filtered_detections


# Rest of your code remains unchanged


def extract_pdf_text(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error("Error: Failed to extract text from PDF. Please upload a valid PDF file.")
        return None

def get_response(user_question, memory, model):
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )

    # Use PDF text as input if available
    if 'pdf_text' in st.session_state:
        conversation_context = st.session_state.pdf_text
        # Append user's original question to the context
        conversation_context += "\n" + user_question
    else:
        conversation_context = user_question

    return conversation(conversation_context)

def submit():
    st.session_state.user_question = st.session_state.widget
    st.session_state.widget = ""


def main():
    initialize_yolo_and_camera()

    global filtered_detections
    st.sidebar.title("Brahman Select an LLM")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['Mixtral-8x7b-32768', 'llama2-70b-4096'],
        key="model_key"  # Unique key for this selectbox
    )

    conversational_memory_length = st.sidebar.slider('Conversational Memory Length:', 1, 10, value=5)
    
    memory = ConversationBufferMemory(k=conversational_memory_length)

    # PDF reading feature
    st.sidebar.header("PDF Reading")
    pdf = st.sidebar.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        pdf_text = extract_pdf_text(pdf)
        if pdf_text:
            st.session_state.pdf_text = pdf_text

    st.title("Brahmmy's Chat and Object Detection")

    # Create layout with two columns
    col1, col2 = st.columns([2, 1])  # Adjust column widths

    # Object detection display in the first column
    with col1:
        st.markdown("## Object Detection")
        detection_button = st.button("Start Object Detection") 
        capture_button = st.button("Capture Objects")  # Moved capture button outside the loop
        video_placeholder = st.empty()

        present_objects = None  # Initialize present_objects

        if detection_button:  # Check if the detection button is clicked
            prev_time = time.time()  # Initialize prev_time here
            while True:
                ret, frame = cap.read()

                if not ret:
                    st.error("Failed to capture video.")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame, local_filtered_detections = detect_objects(frame)  # Get filtered detections

                class_names = [
                    yolo_model.model.names[class_id]
                    for _, _, class_id, _
                    in local_filtered_detections
                ]

                present_objects = ', '.join(class_names) if class_names else None  # Store present objects in a variable

                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time

                frame_pil = Image.fromarray(frame)

                # Display FPS and present objects
                combined_caption = f"FPS: {int(fps)} | Present Objects: {present_objects or 'None'}"
                video_placeholder.image(frame_pil, caption=combined_caption)

                filtered_detections = local_filtered_detections

                print("Present Objects:", present_objects)  # Print the present objects

                # Check if the detection button is still clicked
                if not detection_button:
                    break  # Exit the loop if the button is not clicked anymore
            
        print("Filtered Detections before capture:", filtered_detections)


        if capture_button:
            if filtered_detections:  # Check if objects are detected
                print("Detected objects found.")
                st.write("Captured Objects: ")
                for detection in filtered_detections:
                    class_id = detection[2]
                    class_name = yolo_model.model.names[class_id]
                    st.write(class_name)
                print("Detected objects:", present_objects)  # Print the detected objects
            else:
                print("No objects detected.")
                st.write("No objects detected.")



    # Chatbot interface in the second column    
    with col2:
        st.markdown("Ask me")

            # Pregenerated questions
        pregenerated_questions = [
            "What is this?",
            "How can we use this?",
            "What activities we can do with this?"
        ]

        # Display pregenerated questions
        st.markdown("## Pregenerated Questions")
        for question in pregenerated_questions:
            if st.button(question):
                st.session_state.user_question = question

        if "user_question" not in st.session_state:
            st.session_state.user_question = ""
        
        st.text_input("Ask a question UBian", key="widget", on_change=submit, placeholder="Please enter a statement or question")

        user_question = st.session_state.user_question

        if user_question.strip():  # Check if input is not empty or whitespace
            specific_words = ["bad"]  # List of most used profanity or sexual content
            if any(word.lower() in user_question.lower() for word in specific_words):  # Check if any specific word/phrase is in user's input
                st.warning("Sorry, as Brahmmy a generative AI intended for educational use, I cannot answer any questions related to love, profanity, sexual or adult content.")
            else:
                try:
                    response = get_response(user_question, memory, model)
                    message = {'human': user_question, 'AI': response['response']}
                    st.session_state.chat_history.append(message)
                except Exception as e:
                    st.error("Error: Failed to get response. Please try again.")

        if not user_question.strip():  # Check if input is empty or whitespace
            st.warning("Sorry, I cannot help you without any input information, please don't hesitate to ask, I'm glad to help")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        st.subheader("Chat History")
        for message in reversed(st.session_state.chat_history):
            st.write(f"UBian: {message['human']}")
            st.write(f"Brahmmy: {message['AI']}")

if __name__ == '__main__':
    main()



# import cv2
# import streamlit as st
# from PIL import Image
# from ultralytics import YOLO
# import time
# import supervision as sv
# import os
# from groq import Groq
# from langchain.chains import ConversationChain
# from langchain.chains.conversation.memory import ConversationBufferMemory
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader

# st.set_page_config(layout="wide")

# load_dotenv()

# groq_api_key = os.environ.get('GROQ_API_KEY')

# yolo_model = None
# cap = None

# def initialize_yolo_and_camera():
#     global yolo_model, cap
#     yolo_model = YOLO("best.pt")
#     frame_width, frame_height = 640, 360  # Change as needed
#     cap = cv2.VideoCapture(0) #0 is built-in or first camera and 1 is plug and play
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# def detect_objects(frame):
#     result = yolo_model(frame)[0]
#     detections = sv.Detections.from_yolov8(result)

#     filtered_detections = []
#     for detection in detections:
#         _, confidence, class_id, _ = detection
#         if confidence > 0.7:
#             class_name = yolo_model.model.names[class_id]
#             if class_name in ['Arduino', 'Breadboard']:
#                 filtered_detections.append(detection)

#     labels = [
#         f"{yolo_model.model.names[class_id]} {confidence:0.2f}"
#         for _, confidence, class_id, _
#         in filtered_detections
#     ]

#     for detection, label in zip(filtered_detections, labels):
#         x, y, w, h = map(int, detection[0])
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     return frame, filtered_detections

# def extract_pdf_text(pdf_file):
#     try:
#         pdf_reader = PdfReader(pdf_file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#         return text
#     except Exception as e:
#         st.error("Error: Failed to extract text from PDF. Please upload a valid PDF file.")
#         return None

# def get_response(user_question, memory, model):
#     groq_chat = ChatGroq(
#         groq_api_key=groq_api_key,
#         model_name=model
#     )

#     conversation = ConversationChain(
#         llm=groq_chat,
#         memory=memory
#     )

#     # Use PDF text as input if available
#     if 'pdf_text' in st.session_state:
#         conversation_context = st.session_state.pdf_text
#         # Append user's original question to the context
#         conversation_context += "\n" + user_question
#     else:
#         conversation_context = user_question

#     return conversation(conversation_context)

# def submit():
#     st.session_state.user_question = st.session_state.widget
#     st.session_state.widget = ""

# def main():
#     initialize_yolo_and_camera()
#     st.sidebar.title("Brahman Select an LLM")
#     model = st.sidebar.selectbox(
#         'Choose a model',
#         ['Mixtral-8x7b-32768', 'llama2-70b-4096'],
#         key="model_key"  # Unique key for this selectbox
#     )

#     conversational_memory_length = st.sidebar.slider('Conversational Memory Length:', 1, 10, value=5)
    
#     memory = ConversationBufferMemory(k=conversational_memory_length)

#     # PDF reading feature
#     st.sidebar.header("PDF Reading")
#     pdf = st.sidebar.file_uploader("Upload your PDF", type='pdf')
#     if pdf is not None:
#         pdf_text = extract_pdf_text(pdf)
#         if pdf_text:
#             st.session_state.pdf_text = pdf_text

#     st.title("Brahmmy's Chat and Object Detection")

#     # Create layout with two columns
#     col1, col2 = st.columns([2, 1])  # Adjust column widths

    
    
#     # Object detection display in the first column
#     with col1:
#         st.markdown("## Object Detection")
#         detection_button = st.button("Start Object Detection") 
#         capture_button = st.button("Capture Objects")


#         if detection_button:  # Check if the detection button is clicked
#             video_placeholder = st.empty()

#             prev_time = time.time()  # Initialize prev_time here

#             while True:
#                 ret, frame = cap.read()

#                 if not ret:
#                     st.error("Failed to capture video.")
#                     break

#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame, filtered_detections = detect_objects(frame)

#                 class_names = [
#                     yolo_model.model.names[class_id]
#                     for _, _, class_id, _
#                     in filtered_detections
#                 ]

#                 current_time = time.time()
#                 fps = 1 / (current_time - prev_time)
#                 prev_time = current_time

#                 frame_pil = Image.fromarray(frame)

#                 # Display FPS and present objects
#                 combined_caption = f"FPS: {int(fps)} | Present Objects: {', '.join(class_names)}"
#                 video_placeholder.image(frame_pil, caption=combined_caption)

#                 # Check if the detection button is still clicked
#                 if not detection_button:
#                     break  # Exit the loop if the button is not clicked anymore

#         if capture_button:  
#             if filtered_detections:  # Check if filtered_detections is not empty
#                 class_names = [
#                     yolo_model.model.names[class_id]
#                     for _, _, class_id, _
#                     in filtered_detections
#                 ]
#                 filtered_detections = [', '.join(class_names)]
#                 st.write("Captured Objects: ", filtered_detections[0]) 
#             else:
#                 st.write("No objects detected.")
        

#     # Chatbot interface in the second column
#     with col2:
#         st.markdown("Ask me")

#             # Pregenerated questions
#         pregenerated_questions = [
#             "What is this?",
#             "How can we use this?",
#             "What activities we can do with this?"
#         ]

#         # Display pregenerated questions
#         st.markdown("## Pregenerated Questions")
#         for question in pregenerated_questions:
#             if st.button(question):
#                 st.session_state.user_question = question

#         if "user_question" not in st.session_state:
#             st.session_state.user_question = ""
        
#         st.text_input("Ask a question UBian", key="widget", on_change=submit, placeholder="Please enter a statement or question")

#         user_question = st.session_state.user_question

#         if user_question.strip():  # Check if input is not empty or whitespace
#             specific_words = ["shit", "Iloveyou", "Fuck", "What the F", "What the hell", "Bitch", "bj", "sex", "blowjob", "slut", "swear", "boobs", "tits", "suck", "dick", "make love", "bang", "masturbation", "daks", "tite", "puke", "penis", "gang", "vagina", "facial", "creampie" , "quickie", "gag", "alcohol", "putanginamo", "tangina", "tanga", "bobo", "inutil","dumb","pokpok","usbaw","ulaga", "noob","drugs", "inang", "jakol", "jabol", "ejaculate", "anal", "bdsm", "hoe","handjob"]  # List of most used profanity or sexual content
#             if any(word.lower() in user_question.lower() for word in specific_words):  # Check if any specific word/phrase is in user's input
#                 st.warning("Sorry, as Brahmmy a generative AI intended for educational use, I cannot answer any questions related to love, profanity, sexual or adult content.")
#             else:
#                 try:
#                     response = get_response(user_question, memory, model)
#                     message = {'human': user_question, 'AI': response['response']}
#                     st.session_state.chat_history.append(message)
#                 except Exception as e:
#                     st.error("Error: Failed to get response. Please try again.")

#         if not user_question.strip():  # Check if input is empty or whitespace
#             st.warning("Sorry, I cannot help you without any input information, please don't hesitate to ask, I'm glad to help")

#         if "chat_history" not in st.session_state:
#             st.session_state.chat_history = []

#         # Display chat history
#         st.subheader("Chat History")
#         for message in reversed(st.session_state.chat_history):
#             st.write(f"UBian: {message['human']}")
#             st.write(f"Brahmmy: {message['AI']}")

# if __name__ == '__main__':
#     main()






# import cv2
# import streamlit as st
# from PIL import Image
# from ultralytics import YOLO
# import time
# import supervision as sv
# import os
# from groq import Groq
# from langchain.chains import ConversationChain
# from langchain.chains.conversation.memory import ConversationBufferMemory
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader

# st.set_page_config(layout="wide")

# load_dotenv()

# groq_api_key = os.environ.get('GROQ_API_KEY')

# yolo_model = None
# cap = None

# def initialize_yolo_and_camera():
#     global yolo_model, cap
#     yolo_model = YOLO("best.pt")
#     frame_width, frame_height = 640, 360  # Change as needed
#     cap = cv2.VideoCapture(0) #0 is built-in or first camera and 1 is plug and play
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# def detect_objects(frame):
#     result = yolo_model(frame)[0]
#     detections = sv.Detections.from_yolov8(result)

#     filtered_detections = []
#     for detection in detections:
#         _, confidence, class_id, _ = detection
#         if confidence > 0.7:
#             class_name = yolo_model.model.names[class_id]
#             if class_name in ['Arduino', 'Breadboard']:
#                 filtered_detections.append(detection)

#     labels = [
#         f"{yolo_model.model.names[class_id]} {confidence:0.2f}"
#         for _, confidence, class_id, _
#         in filtered_detections
#     ]

#     for detection, label in zip(filtered_detections, labels):
#         x, y, w, h = map(int, detection[0])
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     return frame, filtered_detections

# def extract_pdf_text(pdf_file):
#     try:
#         pdf_reader = PdfReader(pdf_file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#         return text
#     except Exception as e:
#         st.error("Error: Failed to extract text from PDF. Please upload a valid PDF file.")
#         return None

# def get_response(user_question, memory, model):
#     groq_chat = ChatGroq(
#         groq_api_key=groq_api_key,
#         model_name=model
#     )

#     conversation = ConversationChain(
#         llm=groq_chat,
#         memory=memory
#     )

#     # Use PDF text as input if available
#     if 'pdf_text' in st.session_state:
#         conversation_context = st.session_state.pdf_text
#         # Append user's original question to the context
#         conversation_context += "\n" + user_question
#     else:
#         conversation_context = user_question

#     return conversation(conversation_context)

# def submit():
#     st.session_state.user_question = st.session_state.widget
#     st.session_state.widget = ""

# def main():
#     initialize_yolo_and_camera()
#     st.sidebar.title("Brahman Select an LLM")
#     model = st.sidebar.selectbox(
#         'Choose a model',
#         ['Mixtral-8x7b-32768', 'llama2-70b-4096'],
#         key="model_key"  # Unique key for this selectbox
#     )

#     conversational_memory_length = st.sidebar.slider('Conversational Memory Length:', 1, 10, value=5)
    
#     memory = ConversationBufferMemory(k=conversational_memory_length)

#     # PDF reading feature
#     st.sidebar.header("PDF Reading")
#     pdf = st.sidebar.file_uploader("Upload your PDF", type='pdf')
#     if pdf is not None:
#         pdf_text = extract_pdf_text(pdf)
#         if pdf_text:
#             st.session_state.pdf_text = pdf_text

#     st.title("Brahmmy's Chat and Object Detection")

#     # Create layout with two columns
#     col1, col2 = st.columns([2, 1])  # Adjust column widths

#     # Object detection display in the first column
#     with col1:
#         st.markdown("## Object Detection")
#         detection_button = st.button("Start Object Detection") 

#         if detection_button:  # Check if the detection button is clicked
#             video_placeholder = st.empty()

#             prev_time = time.time()  # Initialize prev_time here

#             while True:
#                 ret, frame = cap.read()

#                 if not ret:
#                     st.error("Failed to capture video.")
#                     break

#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame, filtered_detections = detect_objects(frame)

#                 class_names = [
#                     yolo_model.model.names[class_id]
#                     for _, _, class_id, _
#                     in filtered_detections
#                 ]

#                 current_time = time.time()
#                 fps = 1 / (current_time - prev_time)
#                 prev_time = current_time

#                 frame_pil = Image.fromarray(frame)

#                 # Display FPS and present objects
#                 combined_caption = f"FPS: {int(fps)} | Present Objects: {', '.join(class_names)}"
#                 video_placeholder.image(frame_pil, caption=combined_caption)

#                 # Check if the detection button is still clicked
#                 if not detection_button:
#                     break  # Exit the loop if the button is not clicked anymore

#     # Chatbot interface in the second column
#     with col2:
#         st.markdown("Ask me")

#             # Pregenerated questions
#         pregenerated_questions = [
#             "What is this?",
#             "Who created this?",
#             "What can we do with this?"
#         ]

#         # Display pregenerated questions
#         st.markdown("## Pregenerated Questions")
#         for question in pregenerated_questions:
#             if st.button(question):
#                 st.session_state.user_question = question

#         if "user_question" not in st.session_state:
#             st.session_state.user_question = ""
        
#         st.text_input("Ask a question UBian", key="widget", on_change=submit, placeholder="Please enter a statement or question")

#         user_question = st.session_state.user_question

#         if user_question.strip():  # Check if input is not empty or whitespace
#             specific_words = [""]  # List of most used profanity or sexual content
#             if any(word.lower() in user_question.lower() for word in specific_words):  # Check if any specific word/phrase is in user's input
#                 st.warning("Sorry, as Brahmmy a generative AI intended for educational use, I cannot answer any questions related to love, profanity, sexual or adult content.")
#             else:
#                 try:
#                     response = get_response(user_question, memory, model)
#                     message = {'human': user_question, 'AI': response['response']}
#                     st.session_state.chat_history.append(message)
#                 except Exception as e:
#                     st.error("Error: Failed to get response. Please try again.")

#         if not user_question.strip():  # Check if input is empty or whitespace
#             st.warning("Please enter a statement or question.")

#         if "chat_history" not in st.session_state:
#             st.session_state.chat_history = []

#         # Display chat history
#         st.subheader("Chat History")
#         for message in reversed(st.session_state.chat_history):
#             st.write(f"UBian: {message['human']}")
#             st.write(f"Brahmmy: {message['AI']}")

# if __name__ == '__main__':
#     main()


