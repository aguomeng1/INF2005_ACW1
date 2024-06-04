import cv2
import numpy as np
import wave
import pygame
from tkinter import *
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
import base64
import time

def to_bin(data):
    """Convert `data` to binary format as string"""
    if isinstance(data, str):
        return ''.join([format(ord(i), "08b") for i in data])
    elif isinstance(data, bytes) or isinstance(data, np.ndarray):
        return [format(i, "08b") for i in data]
    elif isinstance(data, int) or isinstance(data, np.uint8):
        return format(data, "08b")
    else:
        raise TypeError("Type not supported.")

#WHITESPACE ENCODING & DECODE
def encode_whitespace(cover_text, payload, marker="<<<hidden>>>"):
    binary_payload = to_bin(payload)
    whitespace_payload = ''.join(' ' if bit == '0' else '\t' for bit in binary_payload)
    stego_text = cover_text + marker + whitespace_payload
    return stego_text

def decode_whitespace(stego_text, marker="<<<hidden>>>"):
    if marker not in stego_text:
        raise ValueError("Marker not found in the text.")
    
    parts = stego_text.split(marker)
    if len(parts) != 2:
        raise ValueError("Incorrect marker usage or multiple markers found.")
    
    whitespace_payload = parts[1]
    binary_payload = ''.join('0' if char == ' ' else '1' for char in whitespace_payload)
    text = ''.join(chr(int(binary_payload[i:i+8], 2)) for i in range(0, len(binary_payload), 8))
    return text

#IMG ENCODE AND DECODE
def encode_image(image_name, secret_data, selected_bits):
    image = cv2.imread(image_name)  # read the image
    n_bytes = (image.shape[0] * image.shape[1] * 3 * selected_bits) // 8  # maximum bytes to encode
    secret_data += "====="  # add stopping criteria
    if len(secret_data) > n_bytes:
        raise ValueError("[!] Insufficient bytes, need bigger image or less data.")

    data_index = 0
    binary_secret_data = to_bin(secret_data)  # convert data to binary
    data_len = len(binary_secret_data)  # size of data to hide

    for row in image:
        for pixel in row:
            r, g, b = to_bin(pixel)  # convert RGB values to binary format
            if data_index < data_len:  # modify the least significant bit only if there is still data to store
                pixel[0] = int(r[:-selected_bits] + binary_secret_data[data_index:data_index+selected_bits][::-1], 2)  # modify from lsb to msb
                data_index += selected_bits
            if data_index < data_len:
                pixel[1] = int(g[:-selected_bits] + binary_secret_data[data_index:data_index+selected_bits][::-1], 2)  # modify from lsb to msb
                data_index += selected_bits
            if data_index < data_len:
                pixel[2] = int(b[:-selected_bits] + binary_secret_data[data_index:data_index+selected_bits][::-1], 2)  # modify from lsb to msb
                data_index += selected_bits
            if data_index >= data_len:  # if data is encoded, just break out of the loop
                break
        if data_index >= data_len:
            break
    return image

def decode_image(image_name, selected_bits):
    image = cv2.imread(image_name)  # read the image
    binary_data = ""
    for row in image:
        for pixel in row:
            r, g, b = to_bin(pixel)
            binary_data += r[-selected_bits:][::-1]  # decode from lsb to msb
            binary_data += g[-selected_bits:][::-1]
            binary_data += b[-selected_bits:][::-1]
    # split by 8-bits
    all_bytes = [binary_data[i: i+8] for i in range(0, len(binary_data), 8)]
    # convert from bits to characters
    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data[-5:] == "=====":
            break
    return decoded_data[:-5]

#AUDIO ENCODE AND DECODE
def encode_audio(audio_name, secret_data, selected_bits):
    audio = wave.open(audio_name, mode='rb')
    frame_bytes = bytearray(list(audio.readframes(audio.getnframes())))

    secret_data += "====="
    n_bytes = len(frame_bytes)
    binary_secret_data = to_bin(secret_data)
    data_len = len(binary_secret_data)

    if data_len > n_bytes * selected_bits:
        raise ValueError("[!] Insufficient bytes, need larger audio file or less data.")

    data_index = 0
    for i in range(len(frame_bytes)):
        for bit in range(selected_bits):
            if data_index < data_len:
                # replace frame with secret data
                frame_bytes[i] = (frame_bytes[i] & ~(1 << bit)) | (int(binary_secret_data[data_index]) << bit)
                data_index += 1
            else:
                break

    encoded_audio_path = f'encoded_audio_{int(time.time())}.wav'
    with wave.open(encoded_audio_path, 'wb') as encoded_audio:
        encoded_audio.setparams(audio.getparams())
        encoded_audio.writeframes(frame_bytes)

    audio.close()
    return encoded_audio_path

def decode_audio(audio_name, selected_bits):
    audio = wave.open(audio_name, mode='rb')
    frame_bytes = bytearray(list(audio.readframes(audio.getnframes())))

    binary_data = ''.join(
        ''.join(str((byte >> bit) & 1) for bit in range(selected_bits))
        for byte in frame_bytes
    )

    all_bytes = [binary_data[i: i+8] for i in range(0, len(binary_data), 8)]
    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data[-5:] == "=====":
            break

    audio.close()
    return decoded_data[:-5]

#IMAGE PAYLOAD ENCODE AND DECODE
def encode_image_with_image(cover_image_path, payload_image_path, selected_bits):
    cover_image = cv2.imread(cover_image_path)  # read the image
    n_bytes = (cover_image.shape[0] * cover_image.shape[1] * 3 * selected_bits) // 8
    print(f"cover image binary data: {n_bytes}")

    # Read the payload image and convert it to binary
    payload_image = cv2.imread(payload_image_path)
    payload_binary_data = ""
    for row in payload_image:
        for pixel in row:
            r, g, b = to_bin(pixel[0]), to_bin(pixel[1]), to_bin(pixel[2])
            payload_binary_data += r + g + b

    # Convert the dimensions to binary and prepend them to the payload data
    height, width, _ = payload_image.shape
    dimension_data = f"{height:016b}{width:016b}"
    payload_binary_data = dimension_data + payload_binary_data

    # Print the payload binary data for comparison
    print(f"Payload binary data: {payload_binary_data[:256]}...")
    
    #Check size
    payload_length = len(payload_binary_data)
    print(f"payload image binary data: {payload_length}")
    if payload_length > n_bytes * 8:
        raise ValueError("[!] Insufficient bytes, need a bigger cover image or smaller payload image.")
    
    data_index = 0
    for row in cover_image:
        for pixel in row:
            r, g, b = to_bin(pixel[0]), to_bin(pixel[1]), to_bin(pixel[2])
            if data_index < payload_length:
                pixel[0] = int(r[:-selected_bits] + payload_binary_data[data_index:data_index+selected_bits], 2)
                data_index += selected_bits
            if data_index < payload_length:
                pixel[1] = int(g[:-selected_bits] + payload_binary_data[data_index:data_index+selected_bits], 2)
                data_index += selected_bits
            if data_index < payload_length:
                pixel[2] = int(b[:-selected_bits] + payload_binary_data[data_index:data_index+selected_bits], 2)
                data_index += selected_bits
            if data_index >= payload_length:
                break
        if data_index >= payload_length:
            break

    return cover_image

def decode_image_with_image(encoded_image_path, selected_bits):
    encoded_image = cv2.imread(encoded_image_path)  # read the encoded image

    binary_data = ""
    for row in encoded_image:
        for pixel in row:
            r, g, b = to_bin(pixel[0]), to_bin(pixel[1]), to_bin(pixel[2])
            binary_data += r[-selected_bits:] + g[-selected_bits:] + b[-selected_bits:]
    
    # Extract the dimensions
    height = int(binary_data[:16], 2)
    width = int(binary_data[16:32], 2)
    payload_length = height * width * 3 * 8
    
    # Extract the payload binary data
    payload_binary_data = binary_data[32:32 + payload_length]
    
    # Convert binary data to bytes
    all_bytes = [payload_binary_data[i:i+8] for i in range(0, len(payload_binary_data), 8)]
    print(f"Payload data: {all_bytes}")

    payload_data = bytearray([int(b, 2) for b in all_bytes])

    # Print the payload data for comparison
    
    
    # Convert the byte array to a numpy array and reshape it to the original payload image shape
    payload_image = np.frombuffer(payload_data, dtype=np.uint8).reshape((height, width, 3))
    
    return payload_image




class SteganographyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Steganography and Steganalysis")
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop)

        self.cover_file = None
        self.payload_file = None
        self.selected_bits = IntVar()
        self.selected_bits.set(1)
        self.decode_image_with_image_toggle = BooleanVar()

        self.create_widgets()
        pygame.init()

    def create_widgets(self):
        Label(self.root, text="Steganography and Steganalysis", font=("Helvetica", 16)).pack(pady=10)

        # Cover object selection
        Button(self.root, text="Select Cover Object", command=self.select_cover_file).pack(pady=5)
        self.cover_label = Label(self.root, text="No file selected")
        self.cover_label.pack(pady=5)

        # Payload selection
        Button(self.root, text="Select Payload", command=self.select_payload_file).pack(pady=5)
        self.payload_label = Label(self.root, text="No file selected")
        self.payload_label.pack(pady=5)

        # Number of LSBs selection
        Label(self.root, text="Number of LSBs:").pack(pady=5)
        Scale(self.root, from_=1, to_=8, orient=HORIZONTAL, variable=self.selected_bits).pack(pady=5)

        # Encode and Decode buttons
        Button(self.root, text="Encode", command=self.encode).pack(pady=5)
        Button(self.root, text="Decode", command=self.decode).pack(pady=5)

        # Toggle for decode image with image payload
        Checkbutton(self.root, text="Decode Image with Image Payload", variable=self.decode_image_with_image_toggle).pack(pady=5)

        # Play Audio buttons
        Button(self.root, text="Play Cover Audio", command=self.play_cover_audio).pack(pady=5)
        Button(self.root, text="Play Current Encoded Audio", command=self.play_encoded_audio).pack(pady=5)

        # Audio Selection
        Button(self.root, text="Select Audio File", command=self.select_audio_file).pack(pady=5)
        self.audio_file = None

        # Play Selected Audio button
        Button(self.root, text="Play Selected Audio", command=self.play_selected_audio).pack(pady=5)

        # Reset button
        Button(self.root, text="Reset", command=self.reset_state).pack(pady=5)

        # Display area for images
        self.image_frame = Frame(self.root)
        self.image_frame.pack(pady=10)

        self.cover_image_label = Label(self.image_frame, text="Cover Image will appear here")
        self.cover_image_label.grid(row=0, column=0, padx=10)

        self.stego_image_label = Label(self.image_frame, text="Stego Image will appear here")
        self.stego_image_label.grid(row=0, column=1, padx=10)

    def select_cover_file(self):
        self.cover_file = filedialog.askopenfilename(title="Select Cover File", filetypes=(("Image files", "*.bmp;*.png;*.gif"), ("Audio files", "*.wav"), ("Text files", "*.txt")))
        self.cover_label.config(text=self.cover_file)

        if self.cover_file.endswith(('bmp', 'png', 'gif')):
            self.display_image(self.cover_file, self.cover_image_label)

    def select_payload_file(self):
        self.payload_file = filedialog.askopenfilename(title="Select Payload File", filetypes=(("Text files", "*.txt"),("Image files", "*.png")))
        self.payload_label.config(text=self.payload_file)

    def encode(self):
        if not self.cover_file or not self.payload_file:
            messagebox.showerror("Error", "Please select both cover and payload files.")
            return

        try:
            if self.cover_file.endswith(('bmp', 'png', 'gif')) and self.payload_file.endswith('png'):
                # Encode image with image payload
                encoded_image = encode_image_with_image(self.cover_file, self.payload_file, self.selected_bits.get())
                encoded_image_path = f'encoded_image_{int(time.time())}.png'
                cv2.imwrite(encoded_image_path, encoded_image)
                self.display_image(encoded_image_path, self.stego_image_label)
                messagebox.showinfo("Success", f"Image payload encoded into image and saved as '{encoded_image_path}'.")
            elif self.cover_file.endswith(('bmp', 'png', 'gif')):
                # Encode image with text payload
                with open(self.payload_file, 'r', encoding='utf-8', errors='ignore') as file:
                    secret_data = file.read()
                encoded_image = encode_image(self.cover_file, secret_data, self.selected_bits.get())
                encoded_image_path = f'encoded_image_{int(time.time())}.png'
                cv2.imwrite(encoded_image_path, encoded_image)
                self.display_image(encoded_image_path, self.stego_image_label)
                messagebox.showinfo("Success", f"Data encoded into image and saved as '{encoded_image_path}'.")
            elif self.cover_file.endswith('wav'):
                # Encode audio with text payload
                with open(self.payload_file, 'r', encoding='utf-8', errors='ignore') as file:
                    secret_data = file.read()
                self.encoded_audio_path = encode_audio(self.cover_file, secret_data, self.selected_bits.get())
                messagebox.showinfo("Success", f"Data encoded into audio and saved as '{self.encoded_audio_path}'.")
            elif self.cover_file.endswith('txt'):
                # Encode text with text payload
                with open(self.cover_file, 'r', encoding='utf-8', errors='ignore') as file:
                    cover_text = file.read()
                with open(self.payload_file, 'r', encoding='utf-8', errors='ignore') as file:
                    secret_data = file.read()
                stego_text = encode_whitespace(cover_text, secret_data)
                encoded_text_path = f'encoded_text_{int(time.time())}.txt'
                with open(encoded_text_path, 'w', encoding='utf-8', errors='ignore') as file:
                    file.write(stego_text)
                messagebox.showinfo("Success", f"Data encoded into text and saved as '{encoded_text_path}'.")
            else:
                messagebox.showerror("Error", "Unsupported file type combination.")
        except Exception as e:
            messagebox.showerror("Error", str(e))


    def reset_state(self):
        self.cover_file = None
        self.payload_file = None
        self.cover_label.config(text="No file selected")
        self.payload_label.config(text="No file selected")
        self.cover_image_label.config(image='', text="Cover Image will appear here")
        self.stego_image_label.config(image='', text="Stego Image will appear here")

    def decode(self):
        if not self.cover_file:
            messagebox.showerror("Error", "Please select a cover file.")
            return

        try:
            if self.cover_file.endswith(('bmp', 'png', 'gif')):
                if self.decode_image_with_image_toggle.get():
                    # Decode image with image payload
                    decoded_image = decode_image_with_image(self.cover_file, self.selected_bits.get())
                    decoded_image_path = f'decoded_payload_image_{int(time.time())}.png'
                    cv2.imwrite(decoded_image_path, decoded_image)
                    self.display_image(decoded_image_path, self.stego_image_label)
                    messagebox.showinfo("Success", f"Image payload decoded from image and saved as '{decoded_image_path}'.")
                else:
                    # Decode image with text payload
                    decoded_data = decode_image(self.cover_file, self.selected_bits.get())
                    messagebox.showinfo("Decoded Data", decoded_data)
                    print(f"Decoded Data: {decoded_data}")
            elif self.cover_file.endswith('wav'):
                decoded_data = decode_audio(self.cover_file, self.selected_bits.get())
                messagebox.showinfo("Decoded Data", decoded_data)
                print(f"Decoded Data: {decoded_data}")
            elif self.cover_file.endswith('txt'):
                with open(self.cover_file, 'r') as file:
                    stego_text = file.read()
                decoded_data = decode_whitespace(stego_text)
                messagebox.showinfo("Decoded Data", decoded_data)
                print(f"Decoded Data: {decoded_data}")
            else:
                raise ValueError("Unsupported file type.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_image(self, image_path, label):
        image = Image.open(image_path)
        image = image.resize((250, 250), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def play_cover_audio(self):
        if not self.cover_file or not self.cover_file.endswith('wav'):
            messagebox.showerror("Error", "Please select a valid cover audio file.")
            return
        pygame.mixer.music.load(self.cover_file)
        pygame.mixer.music.play()

    def play_encoded_audio(self):
        if not self.encoded_audio_path:
            messagebox.showerror("Error", "No encoded audio found.")
            return
        pygame.mixer.music.load(self.encoded_audio_path)
        pygame.mixer.music.play()

    def select_audio_file(self):
        self.audio_file = filedialog.askopenfilename(title="Select Audio File", filetypes=(("Audio files", "*.wav"),))
        if self.audio_file:
            messagebox.showinfo("Success", f"Audio file selected: {self.audio_file}")

    def play_selected_audio(self):
        if not self.audio_file:
            messagebox.showerror("Error", "No audio file selected.")
            return
        pygame.mixer.music.load(self.audio_file)
        pygame.mixer.music.play()


    def drop(self, event):
        dropped_file = event.data
        if dropped_file.endswith(('bmp', 'png', 'gif', 'wav')):
            self.cover_file = dropped_file
            self.cover_label.config(text=self.cover_file)
            if self.cover_file.endswith(('bmp', 'png', 'gif')):
                self.display_image(self.cover_file, self.cover_image_label)
        elif dropped_file.endswith('txt'):
            self.payload_file = dropped_file
            self.payload_label.config(text=self.payload_file)
        else:
            messagebox.showerror("Error", "Unsupported file type.")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = SteganographyApp(root)
    root.mainloop()
