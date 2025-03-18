import hashlib
import re
import tldextract

# Generate a hash signature for an email
def generate_signature(email_content):
    return hashlib.sha256(email_content.encode()).hexdigest()

# Compare email signature against spam signatures
def is_signature_spam(email_content, spam_signatures):
    email_signature = generate_signature(email_content)
    return email_signature in spam_signatures

# Extract links from an email
def extract_links(email_content):
    return re.findall(r'http[s]?://\S+', email_content)

# Check domain's trustworthiness (basic version)
def is_link_trustworthy(link):
    domain = tldextract.extract(link).domain
    trusted_domains = ["example", "google", "yahoo"]  # Add legitimate domains here
    return domain in trusted_domains

# Determine if email is spam based on links
def is_hyperlink_spam(email_content):
    links = extract_links(email_content)
    for link in links:
        if not is_link_trustworthy(link):
            return True
    return False

# Check for unsubscribe links
def has_unsubscribe_link(email_content):
    return "unsubscribe" in email_content.lower()

# Determine if email is spam based on unsubscribe link absence
def is_unsubscribe_spam(email_content):
    return not has_unsubscribe_link(email_content)

# Load spam signatures
def load_spam_signatures(file_path):
    with open(file_path, "r") as f:
        return f.read().splitlines()

# Classify email
def classify_email(email_content, spam_signatures):
    if is_signature_spam(email_content, spam_signatures):
        return "Spam"
    if is_hyperlink_spam(email_content):
        return "Spam"
    if is_unsubscribe_spam(email_content):
        return "Spam"
    return "Not Spam"

# Main program
def main():
    # Load spam signatures
    spam_signatures = load_spam_signatures("spam_signatures.txt")

    # Process email files
    email_files = ["email1.txt", "email2.txt"]  # Add your email file names here
    for email_file in email_files:
        with open(email_file, "r") as f:
            email_content = f.read()
        classification = classify_email(email_content, spam_signatures)
        print(f"Email {email_file}: {classification}")

if __name__ == "__main__":
    main()