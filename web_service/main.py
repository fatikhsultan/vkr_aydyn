from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import xgboost as xgb
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Загрузка модели
model = xgb.Booster()
model.load_model("best_xgb_model.json")


@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def form_post(request: Request):
    form = await request.form()

    def get_checkbox(name: str) -> bool:
        return form.get(name) == "on"

    try:
        to_subway = float(form.get("to_subway", 0))
        kitchen = float(form.get("kitchen", 0))
        floor = float(form.get("floor", 0))
        n_floors = float(form.get("n_floors", 0))
        sqm = float(form.get("sqm", 0))
        km_to_center = float(form.get("km_to_center", 0))

        is_studio_flag = 1 if get_checkbox("is_studio") else 0

        if is_studio_flag:
            n_rooms = 1.0
        else:
            n_rooms = float(form.get("n_rooms", 0))

        material_flags = [
            get_checkbox("is_block"),
            get_checkbox("is_wooden"),
            get_checkbox("is_brick"),
            get_checkbox("is_monolithic_block"),
            get_checkbox("is_monolithic"),
            get_checkbox("is_unknown"),
            get_checkbox("is_panel"),
            get_checkbox("is_stalin")
        ]

        feature_names = [
            "to_subway", "kitchen", "floor", "n_floors", "n_rooms", "sqm",
            "is_block", "is_wooden", "is_brick", "is_monolithic_block", "is_monolithic",
            "is_unknown", "is_panel", "is_stalin", "is_studio", "km_to_center"
        ]

        features = np.array([
            to_subway, kitchen, floor, n_floors, n_rooms, sqm,
            *material_flags, is_studio_flag, km_to_center
        ], dtype=float).reshape(1, -1)

        dmatrix = xgb.DMatrix(features, feature_names=feature_names)
        price_per_sqm = model.predict(dmatrix)[0]
        price_per_sqm_rounded = round(price_per_sqm, 2)
        total_price = round(price_per_sqm * sqm, 2)

        return templates.TemplateResponse("form.html", {
            "request": request,
            "result": price_per_sqm_rounded,
            "total_price": total_price,
            "to_subway": to_subway,
            "kitchen": kitchen,
            "floor": floor,
            "n_floors": n_floors,
            "n_rooms": n_rooms,
            "sqm": sqm,
            "is_block": material_flags[0],
            "is_wooden": material_flags[1],
            "is_brick": material_flags[2],
            "is_monolithic_block": material_flags[3],
            "is_monolithic": material_flags[4],
            "is_unknown": material_flags[5],
            "is_panel": material_flags[6],
            "is_stalin": material_flags[7],
            "is_studio": bool(is_studio_flag),
            "km_to_center": km_to_center
        })

    except Exception as e:
        return templates.TemplateResponse("form.html", {
            "request": request,
            "result": f"Ошибка обработки формы: {e}",
            "total_price": None,
            "to_subway": form.get("to_subway", ""),
            "kitchen": form.get("kitchen", ""),
            "floor": form.get("floor", ""),
            "n_floors": form.get("n_floors", ""),
            "n_rooms": form.get("n_rooms", ""),
            "sqm": form.get("sqm", ""),
            "is_block": form.get("is_block") == "on",
            "is_wooden": form.get("is_wooden") == "on",
            "is_brick": form.get("is_brick") == "on",
            "is_monolithic_block": form.get("is_monolithic_block") == "on",
            "is_monolithic": form.get("is_monolithic") == "on",
            "is_unknown": form.get("is_unknown") == "on",
            "is_panel": form.get("is_panel") == "on",
            "is_stalin": form.get("is_stalin") == "on",
            "is_studio": form.get("is_studio") == "on",
            "km_to_center": form.get("km_to_center", "")
        })


@app.get("/forecast", response_class=HTMLResponse)
async def show_forecast(request: Request):
    return templates.TemplateResponse("forecast.html", {"request": request})