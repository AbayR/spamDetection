function parseMessageContent() {
    const emailBody = document.querySelector('div.ii.gt div[dir="ltr"]');
    if (emailBody) {
        return getTextFromElement(emailBody).trim();
    } else {
        return null;
    }
}

function getTextFromElement(element) {
    let text = '';
    element.childNodes.forEach(node => {
        if (node.nodeType === Node.TEXT_NODE) {
            text += node.textContent;
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            if (node.tagName === 'BR') {
                text += '\n';
            } else if (node.tagName === 'UL' || node.tagName === 'OL') {
                node.querySelectorAll('li').forEach(li => {
                    text += `- ${li.textContent}\n`;
                });
            } else {
                text += getTextFromElement(node);
                if (node.tagName === 'DIV' || node.tagName === 'P') {
                    text += '\n'; // Add newline after div and p tags
                }
            }
        }
    });
    return text;
}

// Listen for messages from the popup script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "parseEmail") {
        const content = parseMessageContent();
        if (content) {
            sendResponse({ content: content });
        } else {
            sendResponse({ content: null });
        }
        return true;  // Keep the message channel open for sendResponse
    }
});
