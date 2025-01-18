from typing import List

import base64
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, CLIPModel
import io
from PIL import Image
import torch
import os

app = FastAPI()

class VLMManager:
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", device_map=self.device)
    
    def identify(self, image: bytes, caption: str) -> List[int]:
        
        im = Image.open(io.BytesIO(image))
        # text prompts
        inputs = self.processor(text=[caption], images=im, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([im.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, threshold=0.1, target_sizes=target_sizes
            )[0]

        bbox = results["boxes"].tolist()
        return bbox