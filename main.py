from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
import seaborn as sns
import statsmodels.api as sm

app = FastAPI()

# Настройка статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Глобальная переменная для хранения данных
current_df = None


def create_regression_plot(x, y, x_label, y_label):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Строим scatter plot с линией регрессии
    plot = sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.5})

    # Рассчитываем параметры регрессии
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Добавляем аннотацию с параметрами
    plt.annotate(
        f'y = {slope:.2f}x + {intercept:.2f}\nR² = {r_value ** 2:.2f}\np-value = {p_value:.2e}',
        xy=(0.05, 0.9),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8)
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Линейная регрессия: {y_label} vs {x_label}')

    # Сохраняем график в буфер
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    global current_df
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        current_df = df

        # Автоматическое определение числовых и датовых колонок
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or
                     any(keyword in col.lower() for keyword in ['date', 'time', 'day'])]

        # Показываем первые и последние 10 строк
        first_10 = df.head(10)
        last_10 = df.tail(10)
        combined = pd.concat([first_10, pd.DataFrame([["..."] * len(df.columns)], columns=df.columns), last_10])

        return templates.TemplateResponse("results.html", {
            "request": request,
            "filename": file.filename,
            "table_html": combined.to_html(classes="table table-striped", index=False),
            "numeric_columns": numeric_cols,
            "date_columns": date_cols,
            "first_rows": first_10.to_dict(orient='records'),
            "last_rows": last_10.to_dict(orient='records')
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })


@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request):
    numeric_cols = [col for col in current_df.columns if pd.api.types.is_numeric_dtype(current_df[col])]
    date_cols = [col for col in current_df.columns if pd.api.types.is_datetime64_any_dtype(current_df[col]) or
                 any(keyword in col.lower() for keyword in ['date', 'time', 'day'])]

    return templates.TemplateResponse("analyze.html", {
        "request": request,
        "numeric_columns": numeric_cols,
        "date_columns": date_cols,
        "has_date_columns": bool(date_cols)
    })


@app.post("/run_t_test", response_class=HTMLResponse)
async def run_t_test(
        request: Request,
        column: str = Form(...),
        date_column: str = Form(None),
        start_date: str = Form(None),
        end_date: str = Form(None),
        reference_value: float = Form(...)
):
    try:
        filtered_data = current_df[column].dropna()

        # Если указаны даты, фильтруем по ним
        if date_column and start_date and end_date:
            if not pd.api.types.is_datetime64_any_dtype(current_df[date_column]):
                current_df[date_column] = pd.to_datetime(current_df[date_column])

            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            mask = (current_df[date_column] >= start_dt) & (current_df[date_column] <= end_dt)
            filtered_data = current_df.loc[mask, column].dropna()

        # Выполняем t-тест
        t_stat, p_value = stats.ttest_1samp(filtered_data, reference_value)

        return templates.TemplateResponse("t_test_results.html", {
            "request": request,
            "column": column,
            "date_column": date_column if date_column else "Не применялось",
            "date_range": f"{start_date} - {end_date}" if date_column and start_date and end_date else "Не применялось",
            "reference_value": reference_value,
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "interpretation": interpret_p_value(p_value),
            "sample_size": len(filtered_data),
            "mean": round(filtered_data.mean(), 4),
            "std_dev": round(filtered_data.std(), 4)
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })


@app.post("/run_regression", response_class=HTMLResponse)
async def run_regression(
        request: Request,
        x_column: str = Form(...),
        y_column: str = Form(...)
):
    try:
        # Удаляем строки с пропущенными значениями
        clean_df = current_df[[x_column, y_column]].dropna()

        if len(clean_df) == 0:
            raise ValueError("Нет данных для анализа после удаления пропущенных значений")

        # Создаем график регрессии
        plot_base64 = create_regression_plot(clean_df[x_column], clean_df[y_column], x_column, y_column)

        # Дополнительная статистика
        corr_coef = clean_df[x_column].corr(clean_df[y_column])

        return templates.TemplateResponse("regression_results.html", {
            "request": request,
            "x_column": x_column,
            "y_column": y_column,
            "plot_base64": plot_base64,
            "correlation": round(corr_coef, 4),
            "sample_size": len(clean_df)
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })


@app.post("/run_polynomial_regression", response_class=HTMLResponse)
async def run_polynomial_regression(request: Request):
    try:
        required_columns = ['Oil_Price', 'Dollar_Index', 'Stavka', 'isp']
        for col in required_columns:
            if col not in current_df.columns:
                raise ValueError(f"Отсутствует необходимая колонка: {col}")

        df = current_df[required_columns].dropna()
        if len(df) < 10:
            raise ValueError("Недостаточно данных для анализа")

        # Создаем матрицу признаков
        X = df[['Oil_Price', 'Dollar_Index', 'Stavka']].copy()
        X['Oil_Price*Dollar_Index'] = X['Oil_Price'] * X['Dollar_Index']
        X['Oil_Price*Stavka'] = X['Oil_Price'] * X['Stavka']
        X['Stavka*Dollar_Index'] = X['Stavka'] * X['Dollar_Index']
        X = sm.add_constant(X)

        y = df['isp']

        # Строим модель
        model = sm.OLS(y, X).fit()

        # Готовим данные для визуализации
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y, y=model.predict(X))
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Фактические значения isp')
        plt.ylabel('Предсказанные значения isp')
        plt.title('Фактические vs Предсказанные значения')

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Формируем таблицу коэффициентов
        coefficients = []
        for name, value, pvalue in zip(model.params.index, model.params, model.pvalues):
            coefficients.append({
                "name": name,
                "coef": round(value, 4),
                "pvalue": f"{pvalue:.4e}"
            })

        return templates.TemplateResponse("polynomial_regression_results.html", {
            "request": request,
            "plot_base64": plot_base64,
            "r_squared": round(model.rsquared, 4),
            "adj_r_squared": round(model.rsquared_adj, 4),
            "coefficients": coefficients,
            "sample_size": len(df)
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })
def interpret_p_value(p_value):
    if p_value < 0.01:
        return "Очень сильные доказательства против нулевой гипотезы (p < 0.01)"
    elif p_value < 0.05:
        return "Сильные доказательства против нулевой гипотезы (p < 0.05)"
    elif p_value < 0.1:
        return "Слабые доказательства против нулевой гипотезы (p < 0.1)"
    else:
        return "Недостаточно доказательств против нулевой гипотезы (p ≥ 0.1)"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)