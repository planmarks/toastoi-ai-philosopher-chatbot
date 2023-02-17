function chatbot_Response() {
  // Display loader while waiting for response
  const loaderElement = $('#loader');
  loader(loaderElement);

  $.ajax({
    type: "POST",
    url: "/chatbot",
    data: { question: $('#message-input').val() },
    success: function(response) {
      // Clear loader
      clearInterval(loadInterval);
      loaderElement.text('');

      // Append user and response message to chat history
      var chatHistory = $('#chat-history');
      chatHistory.append($('<div class="chat-message user-message">').text($('#message-input').val()));
      
      // Display response message with typing animation
      const responseElement = $('<div class="chat-message toastoi-message">');
      chatHistory.append(responseElement);
      typeText(responseElement, response['response']);

      chatHistory.scrollTop(chatHistory[0].scrollHeight);

      // Clear input field
      $('#message-input').val('');
    },
    error: function(xhr, status, error) {
      console.error('Error:', error);
    }
  });
}
