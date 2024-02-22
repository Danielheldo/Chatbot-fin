document.addEventListener("DOMContentLoaded", function() {
    // Füge diese Zeile hinzu, um automatisch zum Seitenende zu scrollen
    scrollToBottom();

    document.getElementById("submitmsg").addEventListener("click", sendMessage);
    document.getElementById("usermsg").addEventListener("keydown", function(event) {
        if (event.keyCode === 13) {
            sendMessage();
        }
    });

    document.getElementById("rating-form").addEventListener("submit", function(event) {
        event.preventDefault();
        var rating = document.getElementById("rating").value;
        fetch("/rate", { 
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ rating: rating })
        })
        .then(response => response.text())
        .then(data => console.log("Bewertung gesendet:", data))
        .catch(error => console.error("Fehler beim Senden der Bewertung:", error));
    });

    function sendMessage() {
        // Automatisches Scrollen zur neuesten Benutzernachricht
        scrollToBottom();
    }
    
    function scrollToBottom() {
        var chatbox = document.getElementById("chatbox");
        chatbox.scrollTop = chatbox.scrollHeight;
    }
    
});
