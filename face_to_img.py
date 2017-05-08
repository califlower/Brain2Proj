import sys
import os
from PIL import Image, ImageDraw
import dlib
from skimage import io
import tempfile

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('detector.dat')

processed_data = open('training/processed.csv', 'w+')

with open('training/data1.csv') as f:
    for l in f:
        img = Image.new('L', (48, 48))
        pixels = [int(x) for x in l.split(',')[1].strip().split(' ')]
        
        for x in range(48):
            for y in range(48):
                img.putpixel((y, x), pixels[x * 48 + y])

        with tempfile.NamedTemporaryFile(suffix='.png') as fp:
            img.save(fp.name)
            img = io.imread(fp.name)
            things = detector(img, 1)
            
            landmark_img = [[255 for x in range(48)] for y in range(48)]
            should_use = False

            for k, d in enumerate(things):
                shape = predictor(img, d)

                for part in range(shape.num_parts):
                    should_use = True
                    point = shape.part(part)
                    landmark_img[abs(int(point.x / 2))][abs(int(point.y / 2))] = 1
            
            if should_use:
                pixels = []

                for row in landmark_img:
                    pixels.extend([str(x) for x in row])

                face_type = l.split(',')[0]
                processed_data.write(face_type + ',' + ' '.join(pixels) + '\n')

processed_data.flush()
processed_data.close()

processed_data = open('testing/processed.csv', 'w+')

with open('testing/data1.csv') as f:
    for l in f:
        img = Image.new('L', (48, 48))
        pixels = [int(x) for x in l.split(',')[1].strip().split(' ')]
        
        for x in range(48):
            for y in range(48):
                img.putpixel((y, x), pixels[x * 48 + y])

        with tempfile.NamedTemporaryFile(suffix='.png') as fp:
            img.save(fp.name)
            img = io.imread(fp.name)
            things = detector(img, 1)
            
            landmark_img = [[255 for x in range(48)] for y in range(48)]
            should_use = False

            for k, d in enumerate(things):
                shape = predictor(img, d)

                for part in range(shape.num_parts):
                    should_use = True
                    point = shape.part(part)
                    landmark_img[abs(int(point.x / 2))][abs(int(point.y / 2))] = 1
            
            if should_use:
                pixels = []

                for row in landmark_img:
                    pixels.extend([str(x) for x in row])

                face_type = l.split(',')[0]
                processed_data.write(face_type + ',' + ' '.join(pixels) + '\n')

processed_data.flush()
processed_data.close()