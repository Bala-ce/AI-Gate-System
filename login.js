document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.querySelector("form");

    if (loginForm) {
        loginForm.addEventListener("submit", async (e) => {
            e.preventDefault();

            const usernameInput = document.getElementById("username").value;
            const passwordInput = document.getElementById("password").value;

            try {
                const response = await fetch("http://localhost:8000/login", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        username: usernameInput,
                        password: passwordInput
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    localStorage.setItem("userRole", data.role);
                    localStorage.setItem("userName", data.name);
                    localStorage.setItem("isLoggedIn", "true");
                    window.location.href = "index.html";
                } else {
                    alert("Invalid username or password!");
                }
            } catch (error) {
                console.error("Login error:", error);
                alert("Could not connect to server. Make sure FastAPI (main.py) is running!");
            }
        });
    }
});
