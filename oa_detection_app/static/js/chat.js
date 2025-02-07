// JavaScript to manage AI Chat Bot interactions
document.addEventListener('DOMContentLoaded', function () {
  const aiBotBtn = document.getElementById('aiBotBtn');
  const chatBox = document.getElementById('chatBox');
  const closeChat = document.getElementById('closeChat');
  const sendChat = document.getElementById('sendChat');
  const chatInput = document.getElementById('chatInput');
  const chatBody = document.getElementById('chatBody');

  if (!aiBotBtn || !chatBox || !closeChat || !sendChat || !chatInput || !chatBody) {
      console.error("Chat elements not found. Please check HTML IDs.");
      return;
  }

  // Toggle chat visibility
  aiBotBtn.addEventListener('click', function () {
      chatBox.style.display = (chatBox.style.display === 'none' || chatBox.style.display === '') ? 'flex' : 'none';
  });

  // Close chat
  closeChat.addEventListener('click', function () {
      chatBox.style.display = 'none';
  });

  // Function to append a message to the chat body
  function appendMessage(sender, text, alignment) {
      const msgDiv = document.createElement('div');
      msgDiv.classList.add(alignment, 'mb-2');
      msgDiv.textContent = `${sender}: ${text}`;
      chatBody.appendChild(msgDiv);
      chatBody.scrollTop = chatBody.scrollHeight; // Auto-scroll
  }

  // Send message function using fetch to call the Flask /chat endpoint
  async function sendMessage() {
      const message = chatInput.value.trim();
      if (message === '') return;

      // Append user's message
      appendMessage("You", message, "text-right");
      chatInput.value = '';
      chatInput.focus();

      try {
          // Prepare form data
          const formData = new FormData();
          formData.append('message', message);

          // Make a POST request to /chat with the custom header to indicate AJAX
          const response = await fetch('/chat', {
              method: 'POST',
              body: formData,
              headers: {
                  'X-Requested-With': 'XMLHttpRequest'
              }
          });

          if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
          }

          // Expect a JSON response in the format: { "response": "..." }
          const data = await response.json();
          appendMessage("Bot", data.response, "text-left");
      } catch (error) {
          console.error("Error sending message:", error);
          appendMessage("Bot", "There was an error processing your message.", "text-left");
      }
  }

  // Listen for send button click
  sendChat.addEventListener('click', sendMessage);

  // Listen for "Enter" key to send message
  chatInput.addEventListener('keypress', function (event) {
      if (event.key === 'Enter') {
          event.preventDefault();
          sendMessage();
      }
  });
});