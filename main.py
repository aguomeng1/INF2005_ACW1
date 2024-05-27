import cv2
import numpy as np
import wave
import pygame
from tkinter import *
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
import base64

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


def encode_image(image_name, secret_data, selected_bits):
    image = cv2.imread(image_name) # read the image
    n_bytes = (image.shape[0] * image.shape[1] * 3 * selected_bits) // 8 # maximum bytes to encode
    secret_data += "=====" # add stopping criteria
    if len(secret_data) > n_bytes:
        raise ValueError("[!] Insufficient bytes, need bigger image or less data.")

    data_index = 0
    binary_secret_data = to_bin(secret_data) # convert data to binary
    data_len = len(binary_secret_data) # size of data to hide

    for row in image:
        for pixel in row:
            r, g, b = to_bin(pixel) # convert RGB values to binary format
            if data_index < data_len:
                pixel[0] = int(r[:-selected_bits] + binary_secret_data[data_index:data_index+selected_bits][::-1], 2)
                data_index += selected_bits
            if data_index < data_len:
                pixel[1] = int(g[:-selected_bits] + binary_secret_data[data_index:data_index+selected_bits][::-1], 2)
                data_index += selected_bits
            if data_index < data_len:
                pixel[2] = int(b[:-selected_bits] + binary_secret_data[data_index:data_index+selected_bits][::-1], 2)
                data_index += selected_bits
            if data_index >= data_len:
                break
        if data_index >= data_len:
            break
    return image

def decode_image(image_name, selected_bits):
    image = cv2.imread(image_name)
    binary_data = ""
    for row in image:
        for pixel in row:
            r, g, b = to_bin(pixel)
            binary_data += r[-selected_bits:][::-1]
            binary_data += g[-selected_bits:][::-1]
            binary_data += b[-selected_bits:][::-1]

    all_bytes = [binary_data[i: i+8] for i in range(0, len(binary_data), 8)]
    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data[-5:] == "=====":
            break
    return decoded_data[:-5]

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
                #replace frame with secret data
                frame_bytes[i] = (frame_bytes[i] & ~(1 << bit)) | (int(binary_secret_data[data_index]) << bit)
                data_index += 1
            else:
                break

    with wave.open('encoded_audio.wav', 'wb') as encoded_audio:
        encoded_audio.setparams(audio.getparams())
        encoded_audio.writeframes(frame_bytes)

    audio.close()

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

        # Play Audio buttons
        Button(self.root, text="Play Cover Audio", command=self.play_cover_audio).pack(pady=5)
        Button(self.root, text="Play Encoded Audio", command=self.play_encoded_audio).pack(pady=5)

        # Display area for images
        self.image_frame = Frame(self.root)
        self.image_frame.pack(pady=10)

        self.cover_image_label = Label(self.image_frame, text="Cover Image will appear here")
        self.cover_image_label.grid(row=0, column=0, padx=10)

        self.stego_image_label = Label(self.image_frame, text="Stego Image will appear here")
        self.stego_image_label.grid(row=0, column=1, padx=10)

    def select_cover_file(self):
        self.cover_file = filedialog.askopenfilename(title="Select Cover File", filetypes=(("Image files", "*.bmp;*.png;*.gif"), ("Audio files", "*.wav")))
        self.cover_label.config(text=self.cover_file)

        if self.cover_file.endswith(('bmp', 'png', 'gif')):
            self.display_image(self.cover_file, self.cover_image_label)

    def select_payload_file(self):
        self.payload_file = filedialog.askopenfilename(title="Select Payload File", filetypes=(("Text files", "*.txt"),))
        self.payload_label.config(text=self.payload_file)

    def encode(self):
        if not self.cover_file or not self.payload_file:
            messagebox.showerror("Error", "Please select both cover and payload files.")
            return

        with open(self.payload_file, 'r') as file:
            secret_data = file.read()

        try:
            if self.cover_file.endswith(('bmp', 'png', 'gif')):
                encoded_image = encode_image(self.cover_file, secret_data, self.selected_bits.get())
                encoded_image_path = 'encoded_image.png'
                cv2.imwrite(encoded_image_path, encoded_image)
                self.display_image(encoded_image_path, self.stego_image_label)
                messagebox.showinfo("Success", "Data encoded into image and saved as 'encoded_image.png'.")
                print(f"Encoded Image Data: {secret_data}")
            elif self.cover_file.endswith('wav'):
                encode_audio(self.cover_file, secret_data, self.selected_bits.get())
                messagebox.showinfo("Success", "Data encoded into audio and saved as 'encoded_audio.wav'.")
                print(f"Encoded Audio Data: {secret_data}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def decode(self):
        if not self.cover_file:
            messagebox.showerror("Error", "Please select a cover file.")
            return

        try:
            if self.cover_file.endswith(('bmp', 'png', 'gif')):
                decoded_data = decode_image(self.cover_file, self.selected_bits.get())
            elif self.cover_file.endswith('wav'):
                decoded_data = decode_audio(self.cover_file, self.selected_bits.get())
            else:
                raise ValueError("Unsupported file type.")
            messagebox.showinfo("Decoded Data", decoded_data)
            print(f"Decoded Data: {decoded_data}")
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
        encoded_audio_path = 'encoded_audio.wav'
        if not encoded_audio_path:
            messagebox.showerror("Error", "No encoded audio found.")
            return
        pygame.mixer.music.load(encoded_audio_path)
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
