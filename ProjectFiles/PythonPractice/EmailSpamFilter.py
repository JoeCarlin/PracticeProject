import imaplib
import email
from email.header import decode_header

# Email credentials
username = "your_email@example.com"
password = "your_password"

# Connect to the server
mail = imaplib.IMAP4_SSL("imap.example.com")

# Login to your account
mail.login(username, password)

# Select the mailbox you want to check
mail.select("inbox")

# Search for all emails in the inbox
status, messages = mail.search(None, "ALL")

# Convert messages to a list of email IDs
email_ids = messages[0].split()

# Define spam keywords
spam_keywords = ["win", "free", "prize", "click here", "subscribe", "buy now"]

# Function to check if an email is spam
def is_spam(email_body):
    for keyword in spam_keywords:
        if keyword in email_body.lower():
            return True
    return False

# Iterate through each email
for email_id in email_ids:
    # Fetch the email by ID
    status, msg_data = mail.fetch(email_id, "(RFC822)")
    
    for response_part in msg_data:
        if isinstance(response_part, tuple):
            # Parse the email content
            msg = email.message_from_bytes(response_part[1])
            email_subject = decode_header(msg["subject"])[0][0]
            email_from = decode_header(msg.get("From"))[0][0]
            
            # If the email message is multipart
            if msg.is_multipart():
                for part in msg.walk():
                    # Extract content type of email
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    
                    try:
                        # Get the email body
                        body = part.get_payload(decode=True).decode()
                    except:
                        pass
                    
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        # Print the plain text part
                        if is_spam(body):
                            print(f"Spam detected from {email_from} with subject {email_subject}")
                            # Move the email to the spam folder
                            mail.store(email_id, '+X-GM-LABELS', '\\Spam')
                        else:
                            print(f"Legitimate email from {email_from} with subject {email_subject}")
            else:
                # If the email message isn't multipart
                content_type = msg.get_content_type()
                body = msg.get_payload(decode=True).decode()
                if content_type == "text/plain":
                    if is_spam(body):
                        print(f"Spam detected from {email_from} with subject {email_subject}")
                        # Move the email to the spam folder
                        mail.store(email_id, '+X-GM-LABELS', '\\Spam')
                    else:
                        print(f"Legitimate email from {email_from} with subject {email_subject}")

# Logout and close the connection
mail.logout()