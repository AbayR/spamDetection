chrome.runtime.onInstalled.addListener(() => {
    console.log('Extension installed');
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    // Handle messages from content scripts or popup here
    if (message.action === "parseEmail") {
        // Logic to handle the message if needed
        console.log('Received message from content script:', message);
    }
    sendResponse({ status: "done" });
});
