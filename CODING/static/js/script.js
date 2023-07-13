$(document).ready(function() {
    $("#submitmsg").click(function() {
        var usermsg = $("#usermsg").val();
        $.get("/get", { msg: usermsg }).done(function(data) {
            var botmsg = $("<div>").text("Bot: " + data).addClass("botmsg");
            $("#chatbox").append(botmsg);
        });
        $("#usermsg").val("");
    });
});
