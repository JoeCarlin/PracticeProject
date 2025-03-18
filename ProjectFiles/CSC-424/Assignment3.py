from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
from Crypto.Random import get_random_bytes
import base64

# Function to generate RSA key pairs (private and public keys)
def generate_rsa_keys():
    key = RSA.generate(2048)  # Generate a 2048-bit RSA key pair
    private_key = key.export_key()  # Export the private key
    public_key = key.publickey().export_key()  # Export the corresponding public key
    return private_key, public_key

# Function to encrypt a message using AES symmetric encryption
def aes_encrypt(plain_text, key):
    cipher = AES.new(key, AES.MODE_EAX)  # Create AES cipher in EAX mode
    ciphertext, tag = cipher.encrypt_and_digest(plain_text.encode())  # Encrypt message
    return base64.b64encode(cipher.nonce + tag + ciphertext).decode()  # Encode to base64

# Function to decrypt a message encrypted with AES
def aes_decrypt(encrypted_text, key):
    encrypted_bytes = base64.b64decode(encrypted_text)  # Decode base64 message
    nonce, tag, ciphertext = encrypted_bytes[:16], encrypted_bytes[16:32], encrypted_bytes[32:]  # Extract nonce, tag, and ciphertext
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)  # Create AES cipher with nonce
    return cipher.decrypt_and_verify(ciphertext, tag).decode()  # Decrypt and verify integrity

# Function to encrypt data using RSA (asymmetric encryption)
def rsa_encrypt(data, public_key):
    rsa_key = RSA.import_key(public_key)  # Import RSA public key
    cipher = PKCS1_OAEP.new(rsa_key)  # Create RSA cipher
    return base64.b64encode(cipher.encrypt(data)).decode()  # Encrypt and encode result to base64

# Function to decrypt RSA-encrypted data using the private key
def rsa_decrypt(encrypted_data, private_key):
    rsa_key = RSA.import_key(private_key)  # Import RSA private key
    cipher = PKCS1_OAEP.new(rsa_key)  # Create RSA cipher
    return cipher.decrypt(base64.b64decode(encrypted_data))  # Decrypt the data

# Function to sign a message using the sender's private key (digital signature)
def sign_message(message, private_key):
    rsa_key = RSA.import_key(private_key)  # Import sender's private key
    hashed_message = SHA256.new(message.encode())  # Hash the message using SHA-256
    signature = pkcs1_15.new(rsa_key).sign(hashed_message)  # Sign the hash
    return base64.b64encode(signature).decode()  # Encode signature to base64

# Function to verify a signature using the sender's public key
def verify_signature(message, signature, public_key):
    rsa_key = RSA.import_key(public_key)  # Import sender's public key
    hashed_message = SHA256.new(message.encode())  # Hash the original message
    try:
        pkcs1_15.new(rsa_key).verify(hashed_message, base64.b64decode(signature))  # Verify the signature
        return True  # Signature is valid
    except (ValueError, TypeError):
        return False  # Signature is invalid

# Function to simulate secure PGP message transmission
def pgp_secure_message(sender_private, sender_public, receiver_private, receiver_public, message):
    print("\n=== Sender Side ===")

    # Step 1: Hash the message (for integrity verification)
    hashed_message = SHA256.new(message.encode()).hexdigest()
    
    # Step 2: Encrypt the hash (digital signature) using sender’s private key
    signature = sign_message(message, sender_private)
    
    # Step 3: Generate a symmetric AES key for encrypting the message
    symmetric_key = get_random_bytes(16)  # 16 bytes = 128-bit key
    
    # Step 4: Encrypt the message using AES and the generated symmetric key
    encrypted_message = aes_encrypt(message, symmetric_key)
    
    # Step 5: Encrypt the AES symmetric key using the receiver's public RSA key
    encrypted_symmetric_key = rsa_encrypt(symmetric_key, receiver_public)

    print("\nEncrypted Message:", encrypted_message)
    print("Encrypted Symmetric Key:", encrypted_symmetric_key)
    print("Digital Signature:", signature)

    print("\n=== Receiver Side ===")

    # Step A: Decrypt the AES symmetric key using receiver’s private RSA key
    decrypted_symmetric_key = rsa_decrypt(encrypted_symmetric_key, receiver_private)
    
    # Step B: Decrypt the message using AES and the obtained symmetric key
    decrypted_message = aes_decrypt(encrypted_message, decrypted_symmetric_key)

    # Step C: Verify the digital signature using sender’s public key
    signature_valid = verify_signature(decrypted_message, signature, sender_public)

    # Step D: Hash the decrypted message
    received_hashed_message = SHA256.new(decrypted_message.encode()).hexdigest()

    # Step E: Compare hashes to verify integrity and authenticity
    if signature_valid and hashed_message == received_hashed_message:
        print("\nMessage Authenticated ✅")
    else:
        print("\nMessage Authentication Failed ❌")

    print("\nDecrypted Message:", decrypted_message)

# Generate RSA keys for sender and receiver
sender_private_key, sender_public_key = generate_rsa_keys()
receiver_private_key, receiver_public_key = generate_rsa_keys()

# Test Case 1
print("\n--- Test Case 1 ---")
pgp_secure_message(sender_private_key, sender_public_key, receiver_private_key, receiver_public_key, "Hello, this is a secure message!")

# Test Case 2
print("\n--- Test Case 2 ---")
pgp_secure_message(sender_private_key, sender_public_key, receiver_private_key, receiver_public_key, "PGP encryption ensures confidentiality and integrity!")