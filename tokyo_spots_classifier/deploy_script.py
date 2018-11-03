from starlette.applications import Starlette
from starlette.responses import JSONResponse
import uvicorn

from fastai import *
from fastai.vision import *

from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio

classes = ['akihabara',
 'imperial_palace',
 'meiji_jingu',
 'odaiba',
 'roppongi_hills',
 'sensoji',
 'shibuya_crossing',
 'shinjuku',
 'shinjuku_gyoen',
 'skytree',
 'tokyo_station',
 'tokyo_tower',
 'tsukiji',
 'ueno']

path=Path('data/lesson1_hw/') 
data = ImageDataBunch.single_from_classes(path,classes,tfms=get_transforms(),size=224)
learn = create_cnn(data, models.resnet34)
learn.load('stage-2')

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

app = Starlette()

@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    img = open_image(BytesIO(bytes))
    pred_class,pred_idx,outputs = learn.predict(img)
    return JSONResponse({
        "predictions": sorted(
            zip(learn.data.classes, map(float, outputs)),
            key=lambda p: p[1],
            reverse=True
            ), "pred_class":pred_class
    })

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8888)
