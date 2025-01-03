import requests

def send_telegram_message(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot7620011385:AAHC3ip1Ha-NeuiTpsvMydRroxIlYJtblro/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Message sent successfully!")
        else:
            print(f"Failed to send message. Response: {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace with your bot token and chat ID
bot_token = "7620011385:AAHC3ip1Ha-NeuiTpsvMydRroxIlYJtblro"  # Replace with your bot token
chat_id = "7396267168"      # Replace with the chat ID
message = "⚠️ Intruder detected! Access denied."

send_telegram_message(bot_token, chat_id, message)
