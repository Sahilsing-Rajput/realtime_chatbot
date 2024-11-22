import os

import asyncio

import pyaudio

import speech_recognition as sr

from openai import AsyncAzureOpenAI

from uuid import uuid4

from realtime import RealtimeClient

from realtime.tools import tools


api_key = "YOUR_API_KEY"

azure_endpoint = "wss://YOUR_AZURE_ENDPOINT"

azure_deployment = "YOUR_AZURE_DEPLOYMENT"


client = AsyncAzureOpenAI(

    api_key=api_key,

    azure_endpoint=azure_endpoint,

    azure_deployment=azure_deployment,

    api_version="2024-10-01-preview"

)


# Global variable to hold the RealtimeClient instance

openai_realtime = None


async def setup_openai_realtime(system_prompt: str):

    """Instantiate and configure the OpenAI Realtime Client"""

    global openai_realtime  # Declare the global variable

    openai_realtime = RealtimeClient(system_prompt=system_prompt)

    track_id = str(uuid4())


    async def handle_conversation_updated(event):

        delta = event.get("delta")

        if delta:

            if 'audio' in delta:

                audio = delta['audio']  # Int16Array, audio added

                play_audio(audio)  # Play the audio response

            if 'transcript' in delta:

                transcript = delta['transcript']  # string, transcript added

                print(f"Transcript: {transcript}")


    openai_realtime.on('conversation.updated', handle_conversation_updated)


    coros = [openai_realtime.add_tool(tool_def, tool_handler) for tool_def, tool_handler in tools]

    await asyncio.gather(*coros)


def play_audio(audio_data):

    """Play audio data using PyAudio."""

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,

                    channels=1,

                    rate=16000,

                    output=True)


    stream.write(audio_data)

    stream.stop_stream()

    stream.close()

    p.terminate()


async def listen_for_audio(recognizer):

    """Listen for audio input and return the recognized text."""

    with sr.Microphone() as source:

        print("Listening...")

        audio = recognizer.listen(source)

        user_input = recognizer.recognize_google(audio, language='en-US')

        return user_input


async def send_user_input_to_openai(user_input, openai_realtime):

    """Send the recognized user input to OpenAI."""

    if openai_realtime and openai_realtime.is_connected():

        await openai_realtime.send_user_message_content([{"type": 'input_text', "text": user_input}])

    else:

        print("Please activate voice mode before sending messages!")


system_prompt = """Provide helpful and empathetic support responses to customer inquiries for ShopMe in Hindi language, addressing their requests, concerns, or feedback professionally.


Maintain a friendly and service-oriented tone throughout the interaction to ensure a positive customer experience.


# Steps


1. **Identify the Issue:** Carefully read the customer's inquiry to understand the problem or question they are presenting.

2. **Gather Relevant Information:** Check for any additional data needed, such as order numbers or account details, while ensuring the privacy and security of the customer's information.

3. **Formulate a Response:** Develop a solution or informative response based on the understanding of the issue. The response should be clear, concise, and address all parts of the customer's concern.

4. **Offer Further Assistance:** Invite the customer to reach out again if they need more help or have additional questions.

5. **Close Politely:** End the conversation with a polite closing statement that reinforces the service commitment of ShopMe.


# Output Format


Provide a clear and concise paragraph addressing the customer's inquiry, including:

- Acknowledgment of their concern

- Suggested solution or response

- Offer for further assistance

- Polite closing


# Notes

- Greet user with Welcome to ShopMe For the first time only

- always speak in Hindi

- Ensure all customer data is handled according to relevant privacy and data protection laws and ShopMe's privacy policy.

- In cases of high sensitivity or complexity, escalate the issue to a human customer support agent.

- Keep responses within a reasonable length to ensure they are easy to read and understand."""


async def initialize_chat():

    """Send a welcome message and set up the OpenAI Realtime Client."""

    print("Hi, Welcome to ShopMe. How can I help you? (Press Ctrl+C to exit)")

    await setup_openai_realtime(system_prompt=system_prompt + "\n\n Customer ID: 12121")


if __name__ == "__main__":

    asyncio.run(initialize_chat())

    recognizer = sr.Recognizer()


    while True:

        try:

            user_input = asyncio.run(listen_for_audio(recognizer))

            asyncio.run(send_user_input_to_openai(user_input, openai_realtime))


        except sr.UnknownValueError:

            print("Sorry, I did not understand that.")

        except sr.RequestError as e:

            print(f"Could not request results; {e}")

        except KeyboardInterrupt:

            print("Exiting...")

            break