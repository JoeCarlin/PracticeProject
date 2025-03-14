import hashlib

# Given SHA1 hashes for users
users = {
    "John12": "9537935389a343a4fbd831b621b00661b91a5172",
    "Sandra19": "576b1b134cb89b0f8a6c4dd1698479a1151b0e63",
    "coolboy29": "5d7196b530fdd24b8faaaecbaa6b08a29daa1304",
    "spider1999": "d6acda01abcfe8afd510b96c1d0a1645ea4c40b8"
}

# Updated 2025 common passwords list
common_passwords = [
    "123456", "123456789", "qwerty", "password", "12345",
    "qwerty123", "1q2w3e", "12345678", "111111", "1234567890"
]

# Function to generate an MD5 hash
def md5_hash(data):
    return hashlib.md5(data.encode()).hexdigest()

# Function to generate a SHA1 hash
def sha1_hash(data):
    return hashlib.sha1(data.encode()).hexdigest()

# Function to crack passwords
def crack_password(user_index, user_hash):
    salt = md5_hash(str(user_index))  # Generate salt using MD5(counter)
    
    for password in common_passwords:
        test_hash = sha1_hash(salt + password)
        if test_hash == user_hash:
            return password
    return None

# Crack passwords for each user
cracked_passwords = {}
for index, (username, sha1_hash_value) in enumerate(users.items(), start=1):
    password = crack_password(index, sha1_hash_value)
    cracked_passwords[username] = password

# Print results
for user, password in cracked_passwords.items():
    print(f"User: {user}, Password: {password if password else 'Not Found'}")