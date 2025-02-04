import imaplib
import email
from email.header import decode_header

# Email credentials (use an App Password for Gmail, not your main password)
username = "joecarlin30@gmail.com"
password = "jncarlin98720"  # Replace with your app password

# Connect to Gmail's IMAP server
mail = imaplib.IMAP4_SSL("imap.gmail.com")

# Login to your account
try:
    mail.login(username, password)
    print("Logged in successfully!")
except imaplib.IMAP4.error:
    print("Failed to login. Please check your credentials.")
    exit()

# Select the mailbox you want to check
mail.select("inbox")

# Search for all emails in the inbox
status, messages = mail.search(None, "ALL")
if status != "OK":
    print("No messages found!")
    exit()

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
    if status != "OK":
        print(f"Failed to fetch email ID {email_id.decode()}")
        continue
    
    for response_part in msg_data:
        if isinstance(response_part, tuple):
            # Parse the email content
            msg = email.message_from_bytes(response_part[1])
            
            # Decode email subject
            subject, encoding = decode_header(msg["subject"])[0]
            if isinstance(subject, bytes):
                # If it's a bytes, decode to string
                subject = subject.decode(encoding if encoding else "utf-8")
            
            # Decode email sender
            from_, encoding = decode_header(msg.get("From"))[0]
            if isinstance(from_, bytes):
                from_ = from_.decode(encoding if encoding else "utf-8")
            
            print(f"Processing email from: {from_}, Subject: {subject}")
            
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
                        body = ""
                    
                    # Check if the content is plain text
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        if is_spam(body):
                            print(f"Spam detected! Moving email to spam folder.")
                            # Move the email to the spam folder
                            mail.store(email_id, '+X-GM-LABELS', '\\Spam')
                        else:
                            print("Legitimate email.")
            else:
                # If the email message isn't multipart
                content_type = msg.get_content_type()
                try:
                    body = msg.get_payload(decode=True).decode()
                except:
                    body = ""
                
                if content_type == "text/plain":
                    if is_spam(body):
                        print(f"Spam detected! Moving email to spam folder.")
                        mail.store(email_id, '+X-GM-LABELS', '\\Spam')
                    else:
                        print("Legitimate email.")

# Logout and close the connection
mail.logout()
print("Logged out successfully!")