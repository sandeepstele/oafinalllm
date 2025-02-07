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

  // Send message function
  function sendMessage() {
      const message = chatInput.value.trim();
      if (message === '') return;

      appendMessage("You", message, "text-right");

      chatInput.value = '';
      chatInput.focus();

      // Simulate bot response or integrate OpenAI API here
      setTimeout(() => getBotResponse(message), 500);
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

  // Function to append a message
  function appendMessage(sender, text, alignment) {
      const msgDiv = document.createElement('div');
      msgDiv.classList.add(alignment, 'mb-2');
      msgDiv.textContent = `${sender}: ${text}`;
      chatBody.appendChild(msgDiv);
      chatBody.scrollTop = chatBody.scrollHeight; // Auto-scroll
  }

  // Simulated AI Bot response function (Replace with OpenAI API call)
  function getBotResponse(userMessage) {
      const botResponses = {
          "hello": "Hi there! How can I assist you?",
          "how are you": "I'm just a bot, but I'm doing great!",
          "bye": "Goodbye! Have a great day!",
      };

      // Check predefined responses or provide a generic one
      const response = botResponses[userMessage.toLowerCase()] || "I'm here to help!";
      
      appendMessage("Bot", response, "text-left");
  }
});