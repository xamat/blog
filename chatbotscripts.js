document.getElementById('closeChatbot').addEventListener('click', function() {
    var chatbot = document.getElementById('chatbot');
    if (chatbot.style.display === 'none') {
        chatbot.style.display = 'block';
        this.textContent = 'X';
        // Position for the open chatbot (inside the iframe)
        this.style.right = 'calc(19px)'; 
        this.style.bottom = 'calc(520px  - 28px)';
    } else {
        chatbot.style.display = 'none';
        this.textContent = 'Open Chatbot';
        // Position for the closed chatbot (bottom of the page)
        this.style.right = '20px';
        this.style.bottom = '20px';
    }
});
