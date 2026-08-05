document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.querySelector("form");

    if (loginForm) {
        loginForm.addEventListener("submit", async (e) => {
            e.preventDefault();

            const usernameInput = document.getElementById("username").value.trim();
            const passwordInput = document.getElementById("password").value;

            const users = {
                "gateadmin": { password: "admin123", role: "ADMIN" },
                "maingate_pc": { password: "gate123", role: "MAIN_GATE" },
                "transport_mgr": { password: "transport123", role: "TRANSPORT" }
            };

            const user = users[usernameInput];

            if (user && user.password === passwordInput) {
                localStorage.setItem("userRole", user.role);
                localStorage.setItem("userName", usernameInput);
                localStorage.setItem("isLoggedIn", "true");
                window.location.href = "index.html";
            } else {
                alert("Invalid username or password!");
            }
        });
    }
});
