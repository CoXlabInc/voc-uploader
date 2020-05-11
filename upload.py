import xml.etree.ElementTree as ET
import base64
import argparse
import aiohttp
import asyncio
import aiofiles

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    img_file = open('VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (year, image_id), 'rb')

    message = '{"image":"'
    message += base64.b64encode(img_file.read()).decode('UTF-8')
    img_file.close()
    message += '","type":"jpg","labels":['
    first = True

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        if first:
            first = False
        else:
            message += ','
        message += '{"name":"%s","x":%f,"y":%f,"w":%f,"h":%f}' % (cls, bb[0], bb[1], bb[2], bb[3])
    message += ']}'
    return message

async def main():
    async with aiohttp.ClientSession() as session:
        for year, image_set in sets:
            async with aiofiles.open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)) as f:
                contents = await f.read()
                image_ids = contents.strip().split()
                for image_id in image_ids:
                    payload = convert_annotation(year, image_id)
                    r = await session.post('%s/api/v1.0/dlt/image'%(args.url), data=payload, headers={'Content-Type': 'application/json', 'Token': args.token}, ssl=False)
                    print('Upload VOCdevkit/VOC%s/JPEGImages/%s.jpg => %d %s' % (year, image_id, r.status, r.text))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VOC uploader for IoT.own')
    parser.add_argument('url', metavar='URL', help='IoT.own server URL')
    parser.add_argument('token', metavar='token', help='Token for IoT.own API')

    args = parser.parse_args()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
