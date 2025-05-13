# DOC_CHAT_BOT
🧭 App Flow Summary

1. PDF Upload
	•	Users select a .pdf file and click [📤 Upload & Embed].
	•	The file is sent to the backend via POST /upload.
	•	A loading message (🔄 Uploading and embedding...) appears.
	•	Once processed, the file input is cleared, and a success or error message is shown for 2 seconds.

2. Chat with Document
	•	Users type a question in the chat input field.
	•	On submitting, the question is:
	•	Added to the chat window as a user message.
	•	Sent via POST /chat as JSON to the backend.
	•	The response is returned and displayed as a bot message, with an avatar.
	•	Users can also click [🧹 Clear] to reset the chat via POST /clear_chat.

3. Add Manual Knowledge
	•	Users type custom knowledge in a textarea and click [🧠 Submit].
	•	The input is sent to POST /add_knowledge as JSON.
	•	On success:
	•	A confirmation message shows up.
	•	The textarea is cleared.
	•	The message fades after 2 seconds.
