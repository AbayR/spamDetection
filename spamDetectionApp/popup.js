document.addEventListener('DOMContentLoaded', function () {
    // Fetch function to communicate with API
    async function query(data, endpoint, method = "POST", contentType = 'application/json') {
        try {
            const token = await getToken();
            const headers = {
                'Content-Type': contentType
            };
            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }

            const body = contentType === 'application/json' ? JSON.stringify(data) : new URLSearchParams(data);

            console.log(`Making request to ${endpoint}`);
            console.log('Method:', method);
            console.log('Headers:', headers);
            console.log('Body:', body);

            const response = await fetch(
                `http://127.0.0.1:8000/${endpoint}`,  // Update to your local endpoint
                {
                    headers: headers,
                    method: method,
                    body: body,
                }
            );

            console.log('Response status:', response.status);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log('Response data:', result);
            return result;
        } catch (error) {
            console.error('Error querying the API:', error);
            throw error;
        }
    }

    // Get stored JWT token from Chrome storage
    async function getToken() {
        return new Promise((resolve, reject) => {
            chrome.storage.local.get(['jwtToken'], (result) => {
                if (chrome.runtime.lastError) {
                    console.error('Error retrieving token:', chrome.runtime.lastError);
                    reject(chrome.runtime.lastError);
                } else {
                    const token = result.jwtToken;
                    console.log('Token retrieved from storage:', token);
                    resolve(token);
                }
            });
        });
    }

    // Store JWT token in Chrome storage
    function setToken(token) {
        chrome.storage.local.set({ jwtToken: token }, () => {
            if (chrome.runtime.lastError) {
                console.error('Error setting token:', chrome.runtime.lastError);
            } else {
                console.log('Token stored in storage:', token);
            }
        });
    }

    // Remove JWT token from Chrome storage
    function removeToken() {
        chrome.storage.local.remove('jwtToken', () => {
            if (chrome.runtime.lastError) {
                console.error('Error removing token:', chrome.runtime.lastError);
            } else {
                console.log('Token removed from storage');
            }
        });
    }

    // Check if the user is logged in
    async function checkLoginStatus() {
        try {
            const token = await getToken();
            if (token) {
                // Assume a successful token validation if token exists
                document.getElementById('auth-container').style.display = 'none';
                document.getElementById('logged-in-container').style.display = 'block';
                document.getElementById('parse-button-public').style.display = 'none';
            } else {
                document.getElementById('auth-container').style.display = 'block';
                document.getElementById('logged-in-container').style.display = 'none';
                document.getElementById('parse-button-public').style.display = 'block';
            }
        } catch (error) {
            console.error('Error checking login status:', error);
        }
    }

    // Login function
    async function login(email, password) {
        try {
            const response = await query({ username: email, password: password }, 'token', "POST", 'application/x-www-form-urlencoded');
            if (response.access_token) {
                setToken(response.access_token);  // Store the token
                document.getElementById('status').innerText = 'Logged in successfully';
                checkLoginStatus();
            } else {
                document.getElementById('status').innerText = 'Login failed';
            }
        } catch (error) {
            document.getElementById('status').innerText = 'Login failed due to an error';
            console.error('Login error:', error);
        }
    }

    // Registration function
    async function register(email, password) {
        try {
            const response = await query({ email, password }, 'register', "POST", 'application/json');
            if (response.msg) {
                document.getElementById('status').innerText = 'Registered successfully';
            } else {
                document.getElementById('status').innerText = 'Registration failed';
            }
        } catch (error) {
            document.getElementById('status').innerText = 'Registration failed due to an error';
            console.error('Registration error:', error);
        }
    }

    // Add custom rule function
    async function addCustomRule(rule) {
        const response = await query({ rule: rule }, 'custom_rules', "POST");
        if (response.status === "rule added") {
            document.getElementById('status').innerText = 'Custom rule added successfully';
        } else {
            document.getElementById('status').innerText = 'Failed to add custom rule';
        }
    }

    document.getElementById('login-button').addEventListener('click', () => {
        const email = document.getElementById('login-email').value;
        const password = document.getElementById('login-password').value;
        login(email, password);
    });

    document.getElementById('register-button').addEventListener('click', () => {
        const email = document.getElementById('register-email').value;
        const password = document.getElementById('register-password').value;
        register(email, password);
    });

    document.getElementById('parse-button').addEventListener('click', () => {
        chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
            chrome.tabs.sendMessage(tabs[0].id, { action: "parseEmail" }, async (response) => {
                const emailContentElement = document.getElementById('email-content');
                if (response && response.content) {
                    emailContentElement.innerText = response.content;

                    const token = await getToken();
                    const endpoint = token ? 'predict' : 'predict_public';
                    query({ text: response.content }, endpoint, "POST")
                        .then(data => {
                            if (data.prediction === 'spam') {
                                emailContentElement.innerHTML += `<div class="spam">Spam Detection Result: This is a spam message!</div>`;
                            } else {
                                emailContentElement.innerHTML += `<div class="not-spam">Spam Detection Result: This is not a spam message!</div>`;
                            }
                        })
                        .catch(error => {
                            console.error('Error querying the API:', error);
                            emailContentElement.innerText += '\n\nSpam detection failed.';
                        });
                } else {
                    emailContentElement.innerText = 'No content found or failed to parse email.';
                }
            });
        });
    });

    document.getElementById('parse-button-public').addEventListener('click', () => {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            chrome.tabs.sendMessage(tabs[0].id, { action: "parseEmail" }, (response) => {
                const emailContentElement = document.getElementById('email-content');
                if (response && response.content) {
                    emailContentElement.innerText = response.content;

                    query({ text: response.content }, 'predict_public', "POST")
                        .then(data => {
                            if (data.prediction === 'spam') {
                                emailContentElement.innerHTML += `<div class="spam">Spam Detection Result: This is a spam message!</div>`;
                            } else {
                                emailContentElement.innerHTML += `<div class="not-spam">Spam Detection Result: This is not a spam message!</div>`;
                            }
                        })
                        .catch(error => {
                            console.error('Error querying the API:', error);
                            emailContentElement.innerText += '\n\nSpam detection failed.';
                        });
                } else {
                    emailContentElement.innerText = 'No content found or failed to parse email.';
                }
            });
        });
    });

    document.getElementById('go-website-button').addEventListener('click', async () => {
        const token = await getToken();
        chrome.tabs.create({ url: `http://127.0.0.1:8000/report/` });  // Update to your website URL
    });

    document.getElementById('logout-button').addEventListener('click', () => {
        removeToken();
        checkLoginStatus();
        document.getElementById('status').innerText = 'Logged out successfully';
    });

    document.getElementById('add-rule-button').addEventListener('click', () => {
        const rule = document.getElementById('custom-rule-input').value;
        addCustomRule(rule);
    });

    checkLoginStatus();
});
