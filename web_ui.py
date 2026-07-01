import os
import gradio as gr
import warnings
import time
from transformers import pipeline
import speech_recognition as sr
from main import get_response , start_tts
import json
import threading
import logging
import idle_animation
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, module="gradio")

recognizer = sr.Recognizer()
mic = sr.Microphone()
transcriber = pipeline("automatic-speech-recognition", model="UsefulSensors/moonshine-base", device=0)

css = """
#mic-button {
    height: 80px;
    font-size: 20px;
}

"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(c100="#fce7f3", c200="#fbcfe8", c300="#f9a8d4", c400="#f472b6", c50="#f04370", c500="#ec4899", c600="#db2777", c700="#be185d", c800="#9d174d", c900="#831843", c950="#6e1a3d"),
    secondary_hue=gr.themes.Color(c100="#e0e7ff", c200="#c7d2fe", c300="#a5b4fc", c400="#818cf8", c50="#f04370", c500="#6366f1", c600="#4f46e5", c700="#4338ca", c800="#3730a3", c900="#312e81", c950="#2b2c5e"),
    neutral_hue=gr.themes.Color(c100="#f3f4f6", c200="#e5e7eb", c300="#d1d5db", c400="#9ca3af", c50="#6a6289", c500="#6b7280", c600="#4b5563", c700="#374151", c800="#1f2937", c900="#111827", c950="#0b0f19"),
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=[gr.themes.GoogleFont('JetBrains Mono'), 'ui-monospace', 'Consolas', 'monospace'],
).set(
    body_background_fill='*checkbox_background_color',
    body_background_fill_dark='#0c0a18',
    body_text_color='*color_accent_soft',
    body_text_size='*text_lg',
    background_fill_primary_dark='#141025',
    background_fill_secondary_dark='#1a1530',
    shadow_drop_lg='*button_secondary_shadow',
    block_background_fill_dark='#161230',
    block_border_color='*neutral_200',
    block_border_color_dark='#5b21b630',
    block_border_width='1px',
    block_border_width_dark='1px',
    block_label_background_fill='*neutral_100',
    block_label_background_fill_dark='#1e1838',
    block_label_text_weight='500',
    block_shadow_dark='0 2px 12px -2px rgba(14, 12, 30, 0.5), 0 0 0 1px #5b21b615',
    block_title_background_fill='*neutral_100',
    block_title_background_fill_dark='#1e1838',
    block_title_text_color='*neutral_600',
    block_title_text_color_dark='*neutral_300',
    block_title_text_weight='500',
    button_border_width='1px',
    button_primary_background_fill='linear-gradient(135deg, #f04370 0%, #d9245a 100%)',
    button_primary_background_fill_dark='linear-gradient(135deg, #d9245a 0%, #6d28d9 100%)',
    button_primary_background_fill_hover='linear-gradient(135deg, #f86e90 0%, #f04370 100%)',
    button_primary_background_fill_hover_dark='linear-gradient(135deg, #f04370 0%, #7c3aed 100%)',
    button_primary_border_color='#d9245a',
    button_primary_border_color_dark='#f0437040',
    button_primary_shadow='0 1px 3px #f0437030',
    button_primary_shadow_active='none',
    button_primary_shadow_dark='0 2px 12px -2px #d9245a50',
    button_primary_shadow_hover_dark='0 4px 20px -2px #f0437060, 0 0 0 1px #f0437030',
    button_secondary_background_fill_dark='#1e1838',
    button_secondary_background_fill_hover_dark='#261f45',
    button_secondary_border_color_dark='#7c3aed30',
    button_secondary_shadow_dark='0 1px 6px -1px rgba(14, 12, 30, 0.4)',
    button_secondary_shadow_hover_dark='0 4px 12px -2px rgba(14, 12, 30, 0.5)',
    button_cancel_background_fill='white',
    button_cancel_background_fill_dark='#1a1530',
    button_cancel_background_fill_hover='*primary_50',
    button_cancel_background_fill_hover_dark='#7f1d1d25',
    button_cancel_border_color='#fecaca',
    button_cancel_border_color_dark='#b91c1c40',
    button_cancel_text_color='#dc2626',
    button_cancel_text_color_dark='#f87171',
    chatbot_text_size= "*text_lg",
    checkbox_label_background_fill_selected_dark= "linear-gradient(135deg, #b81c4a 0%, #5b21b6 100%)",
)


with gr.Blocks(title="Nova") as demo:

    chatbox = gr.Chatbot(label="Conversation", height=400)

    with gr.Row():

        def send_message(message, history):
            if message:
                history.append({"role": "user", "content": message})
                return message, history
            return "", history
        
        def nova_response(message, history):
            if message:
                nova_reply = get_response(message)
                threading.Thread(target=start_tts, args=(nova_reply["response"], nova_reply["emotion"]), daemon=True).start()
                history.append({"role": "assistant", "content": nova_reply["response"]})
                return "", history
            else:
                try:
                    nova_reply = get_response(history[-1]["content"][0]["text"])
                except Exception as e:
                    print(e)
                    gr.Warning("Error occurred while generating response. Please try again.")
                    return "", history
                threading.Thread(target=start_tts, args=(nova_reply["response"], nova_reply["emotion"]), daemon=True).start()
                history.append({"role": "assistant", "content": nova_reply["response"]})
                return "", history

        user_input = gr.Textbox(label="Your Message", placeholder="Type your message here...")

        user_input.submit(
            fn=send_message,
            inputs=[user_input, chatbox],
            outputs=[user_input, chatbox]
        ).then(fn=nova_response, inputs=[user_input, chatbox], outputs=[user_input, chatbox])
    
    with gr.Row():


        def record_audio():
            record = False
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                try:
                    audio_data = recognizer.listen(source)
                    record = True
                except Exception as e:
                    print(e)

            if record and audio_data is not None:
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio_data.get_wav_data())
            else:
                return "No Audio Captured, Please try Again."
        
        def transcribe_audio():
            try:
                result = transcriber("temp_audio.wav")
                return result['text']
            except Exception as e:
                print("No audio captured.")
                return "Error occurred while transcribing audio. Please try again."

        def start_recording(mic_button):

            if mic_button == "Toggle Mic":
                return "Untoggle Mic"
            else:
                return "Toggle Mic"
        
        def handle_recording(mic_button, history):

            if mic_button == "Untoggle Mic":
                result = record_audio()
                if result == "No Audio Captured, Please try Again.":
                    gr.Warning(result)
                    return "Toggle Mic", history

                transcribed_text = transcribe_audio()
                if "No speech detected" in transcribed_text or "Error occurred" in transcribed_text:
                    gr.Warning(transcribed_text)
                    return "Toggle Mic", history

                history.append({"role": "user", "content": transcribed_text})
                return "Toggle Mic", history
            else:
                return "Toggle Mic", history

        mic_button = gr.Button("Toggle Mic", elem_id="mic-button", variant="primary")

        mic_button.click(
            fn=start_recording,
            inputs=[mic_button],
            outputs=[mic_button],
            ).then(
            fn=handle_recording,
            inputs=[mic_button, chatbox],
            outputs=[mic_button, chatbox]
            ).then(
            fn=nova_response,
            inputs=[user_input, chatbox],
            outputs=[user_input, chatbox]
        )

if __name__ == "__main__":
    with open("config.json", "w") as f:
        json.dump({"WEB_UI_MODE": "ON"}, f)
    idle_animation.start_idle_animation()
    demo.launch(theme=theme, css=css)
