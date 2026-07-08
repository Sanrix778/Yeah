import base64
import os
import re
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components


def normalize_str(value):
    if not isinstance(value, str):
        return ''
    value = value.strip().lower()
    value = unicodedata.normalize('NFKD', value)
    value = ''.join(ch for ch in value if not unicodedata.combining(ch))
    return value


def find_column(columns, pattern):
    pattern = normalize_str(pattern)
    for c in columns:
        if pattern in normalize_str(c):
            return c
    return None


def parse_export_destinations(series, top_n=8):
    counts = {}
    for value in series.dropna().astype(str):
        cleaned = re.sub(r'\(.*?\)', '', value)
        parts = re.split(r'[;,/]', cleaned)
        for part in parts:
            part = part.strip()
            if len(part) < 3:
                continue
            norm = normalize_str(part)
            if any(stop_word in norm for stop_word in ['mercado', 'refineria', 'porcentaje', 'paises', 'proyecto', 'convenio', 'esquema', 'operacion', 'control', 'acuerdo', 'empresa', 'estatal', 'otros', 'mayoria', 'activos', 'energia', 'acciones', 'construccion']):
                continue
            counts[part] = counts.get(part, 0) + 1
    if not counts:
        return []
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]


def load_and_clean(path):
    df = pd.read_csv(path, engine='python', encoding='utf-8', on_bad_lines='skip')
    # Limpiar espacios en los nombres de las columnas
    df.columns = [c.strip() for c in df.columns]

    year_col = (
        find_column(df.columns, 'year')
        or find_column(df.columns, 'ano')
        or find_column(df.columns, 'anio')
        or find_column(df.columns, 'año')
    )
    valor_col = find_column(df.columns, 'valor') or find_column(df.columns, 'value')
    categoria_col = find_column(df.columns, 'categoria') or find_column(df.columns, 'category')
    reservas_col = find_column(df.columns, 'reservas')
    reservas_global_col = (
        find_column(df.columns, 'reservas global')
        or find_column(df.columns, 'reservas globales')
        or find_column(df.columns, 'reservasglobales')
    )

    # Convertir a los tipos de datos correctos
    use_cols = {}
    if year_col:
        use_cols['year'] = year_col
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    if valor_col:
        use_cols['valor'] = valor_col
        df[valor_col] = pd.to_numeric(df[valor_col], errors='coerce')
    if categoria_col:
        use_cols['categoria'] = categoria_col
        df[categoria_col] = df[categoria_col].astype(str)
    if reservas_col:
        df[reservas_col] = pd.to_numeric(df[reservas_col], errors='coerce')
    if reservas_global_col:
        df[reservas_global_col] = pd.to_numeric(df[reservas_global_col], errors='coerce')

    return df, use_cols, reservas_col, reservas_global_col


def build_plots(df, use_cols, reservas_col, reservas_global_col):
    figs = {}
    categoria_col = use_cols.get('categoria')
    valor_col = use_cols.get('valor')
    year_col = use_cols.get('year')

    # Pie: Venezuela reservas vs resto del mundo
    v_res, g_res = None, None
    if reservas_col and reservas_global_col and year_col:
        tmp = df[[year_col, reservas_col, reservas_global_col]].dropna()
        if not tmp.empty:
            latest = tmp.sort_values(year_col).iloc[-1]
            v_res, g_res = float(latest[reservas_col]), float(latest[reservas_global_col])
        else:
            if df[reservas_col].dropna().any():
                v_res = float(df[reservas_col].dropna().iloc[-1])
            if df[reservas_global_col].dropna().any():
                g_res = float(df[reservas_global_col].dropna().iloc[-1])

    if v_res is not None and g_res is not None and g_res > 0:
        rest = max(g_res - v_res, 0.0)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.pie([v_res, rest], labels=['Venezuela', 'Resto del mundo'], autopct='%1.1f%%', startangle=140, colors=['#2b83ba', '#fdae61'])
        ax1.set_title('Reservas: Venezuela vs Resto del Mundo')
        figs['pie_reservas'] = fig1

    # Producción histórica (Filtro estricto por BPD y gráfica de línea)
    prod_bpd = pd.DataFrame()
    if categoria_col and valor_col and year_col and 'unidades' in df.columns:
        mask_prod = (
            df[categoria_col].str.lower().str.contains('producc', na=False) &
            df['unidades'].astype(str).str.lower().str.contains('bpd', na=False)
        )
        prod_bpd = df.loc[mask_prod].dropna(subset=[year_col, valor_col])
        
        if not prod_bpd.empty:
            byyear = prod_bpd.groupby(year_col)[valor_col].sum().sort_index()
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(byyear.index.astype(int), byyear.values, marker='o', color='#4c72b0', linewidth=2)
            ax2.set_xlabel('Año')
            ax2.set_ylabel('Barriles por día (BPD)')
            ax2.set_title('Histórico de Producción (BPD)')
            ax2.grid(True, linestyle='--', alpha=0.6)
            fig2.tight_layout()
            figs['line_produccion'] = fig2

    # Exportaciones históricas (Filtro estricto por BPD y gráfica de línea)
    export_bpd = pd.DataFrame()
    if categoria_col and valor_col and year_col and 'unidades' in df.columns:
        mask_export = (
            df[categoria_col].str.lower().str.contains('exportaci', na=False) &
            ~df[categoria_col].str.lower().str.contains('finanzas', na=False) &
            df['unidades'].astype(str).str.lower().str.contains('bpd', na=False)
        )
        export_bpd = df.loc[mask_export].dropna(subset=[year_col, valor_col])
        
        if not export_bpd.empty:
            byyear_export = export_bpd.groupby(year_col)[valor_col].sum().sort_index()
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(byyear_export.index.astype(int), byyear_export.values, marker='s', color='#ff7f0e', linewidth=2)
            ax3.set_xlabel('Año')
            ax3.set_ylabel('Barriles por día (BPD)')
            ax3.set_title('Histórico de Exportaciones (BPD)')
            ax3.grid(True, linestyle='--', alpha=0.6)
            fig3.tight_layout()
            figs['line_exportaciones'] = fig3

    # Comparativa Producción vs Exportación (BPD)
    if not prod_bpd.empty and not export_bpd.empty:
        combined = pd.DataFrame({
            'Producción': prod_bpd.groupby(year_col)[valor_col].sum(),
            'Exportaciones': export_bpd.groupby(year_col)[valor_col].sum(),
        }).fillna(0).sort_index()
        
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.plot(combined.index.astype(int), combined['Producción'], marker='o', label='Producción', color='#4c72b0', linewidth=2)
        ax4.plot(combined.index.astype(int), combined['Exportaciones'], marker='s', label='Exportaciones', color='#ff7f0e', linewidth=2)
        ax4.set_xlabel('Año')
        ax4.set_ylabel('Barriles por día (BPD)')
        ax4.set_title('Producción vs Exportaciones por año (BPD)')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.6)
        fig4.tight_layout()
        figs['line_produccion_export'] = fig4

    # Destinos de exportación (CORREGIDO)
    if 'distribucion' in df.columns:
        # Filtramos cualquier fila que tenga datos de distribución válidos
        mask_dest = df['distribucion'].notna() & (df['distribucion'].astype(str).str.len() > 5)
        dest_df = df.loc[mask_dest]

        dest_counts = parse_export_destinations(dest_df['distribucion'])
        if dest_counts:
            labels, values = zip(*dest_counts)
            fig5, ax5 = plt.subplots(figsize=(8, 4))
            ax5.barh(labels[::-1], values[::-1], color='#66c2a5')
            ax5.set_xlabel('Frecuencia de menciones en la historia')
            ax5.set_title('Destinos más frecuentes (Todos los periodos)')
            fig5.tight_layout()
            figs['bar_export_destinos'] = fig5

    return figs


def main():
    st.set_page_config(page_title='Análisis Petróleo Venezuela', layout='wide')

    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, 'venezuela_petroleo.csv')

    st.title('Análisis histórico del petróleo de Venezuela')
    st.write('Carga y análisis del dataset `venezuela_petroleo.csv` con Pandas, visualizaciones en Matplotlib y UI con Streamlit.')

    info_col, image_col = st.columns([4, 1])
    with info_col:
        st.write('Hecho por Santiago Hernandez y ChatGPT')
    with image_col:
        image_path = os.path.join(base, 'Cat whit Sanrixluche (11).jpg')
        sounds_dir = os.path.join(base, 'sounds', 'meow')
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode()

            # Collect available meow audio files and encode
            audio_b64_list = []
            if os.path.isdir(sounds_dir):
                for fname in sorted(os.listdir(sounds_dir)):
                    if fname.lower().endswith('.mp3'):
                        try:
                            with open(os.path.join(sounds_dir, fname), 'rb') as af:
                                audio_b64_list.append(base64.b64encode(af.read()).decode())
                        except Exception:
                            continue

            audio_js_array = ','.join([f"'data:audio/mpeg;base64,{s}'" for s in audio_b64_list])

            html_template = """
                <style>
                    .hover-img {
                        transition: transform 0.12s ease;
                        border-radius: 12px;
                        cursor: pointer;
                        display:block;
                        margin-left:auto;
                        margin-right:auto;
                        position:relative;
                        z-index:3;
                    }
                    .hover-img:active {
                        transform: scale(0.92);
                    }
                    /* Texto flotante: elemento separado del <img>, con posición fija
                       para que pueda moverse libremente por la ventana del componente. */
                    .miau-float {
                        position:fixed;
                        font-size:40px;
                        font-family: "Comic Sans MS", "Comic Sans", cursive;
                        font-weight:700;
                        color:#000000;
                        pointer-events:none;
                        transform-origin:center center;
                        z-index:9999; /* sobre la imagen */
                        white-space:nowrap;
                        text-shadow: 0 2px 6px rgba(0,0,0,0.18);
                    }
                </style>
                <img id="catimg" class="hover-img" src="data:image/jpeg;base64,{IMG_B64}" width="120" title="Si, soy yo. Felicidades" />
                <script>
                    const audios = [{AUDIO_ARRAY}];
                    const el = document.getElementById('catimg');
                    function playRandom(){
                        if(!audios || audios.length===0) return;
                        const idx = Math.floor(Math.random()*audios.length);
                        const a = new Audio(audios[idx]);
                        a.play().catch(()=>{});
                    }

                    function spawnMiauAt(rect){
                        const m = document.createElement('div');
                        m.className = 'miau-float';
                        m.textContent = 'Miau';

                        // Start near the image: center X plus a small random offset
                        const startX = rect.left + rect.width/2 + (Math.random()*rect.width - rect.width/2);
                        const startY = rect.top + rect.height/2 + (Math.random()*20 - 10);
                        m.style.left = (startX) + 'px';
                        m.style.top = (startY) + 'px';
                        m.style.transform = 'translate(-50%, -50%) rotate(0deg)';
                        document.body.appendChild(m);

                        // Salto hacia arriba
                        const upKeyframes = [
                            { transform: 'translate(-50%, -50%) translateY(0px) rotate(0deg)', offset:0 },
                            { transform: 'translate(-50%, -50%) translateY(-180px) rotate(0deg)', offset:1 }
                        ];
                        const upTiming = { duration: 500 + Math.random()*200, easing: 'cubic-bezier(.2,.8,.2,1)', fill: 'forwards' };
                        m.animate(upKeyframes, upTiming);

                        // Caída diagonal fuera de la pantalla
                        setTimeout(()=>{
                            const dir = Math.random() < 0.5 ? -1 : 1;
                            const targetX = dir > 0 ? window.innerWidth + 200 : -200;
                            const fallY = window.innerHeight + 200 - startY;
                            const fallKeyframes = [
                                { transform: 'translate(-50%, -50%) translateY(-180px) rotate(0deg)', offset:0 },
                                { transform: `translate(${targetX - startX}px, ${fallY}px) rotate(${dir * 90}deg)`, offset:1 }
                            ];
                            const fallTiming = { duration: 2200 + Math.random()*800, easing: 'cubic-bezier(.2,.6,.1,1)', fill: 'forwards' };
                            const anim = m.animate(fallKeyframes, fallTiming);
                            anim.onfinish = ()=>{ try{ m.remove(); }catch(e){} };
                        }, 520 + Math.random()*200);
                    }

                    el && el.addEventListener('click', ()=>{
                        // pressed effect
                        el.style.transform = 'scale(0.92)';
                        setTimeout(()=>{ el.style.transform = 'scale(1)'; }, 120);
                        playRandom();
                        const rect = el.getBoundingClientRect();
                        spawnMiauAt(rect);
                    });
                </script>
            """

            html = html_template.replace('{IMG_B64}', img_b64).replace('{AUDIO_ARRAY}', audio_js_array)
            components.html(html, height=150)
        else:
            st.write('Imagen no disponible')

    if not os.path.exists(csv_path):
        st.error(f'No se encontró el archivo {csv_path}. Coloca `venezuela_petroleo.csv` en la misma carpeta que esta aplicación.')
        return

    with st.spinner('Cargando y limpiando datos...'):
        df, use_cols, reservas_col, reservas_global_col = load_and_clean(csv_path)

    st.sidebar.header('Filtros')
    st.sidebar.write('Previsualización y datos base')

    st.header('Datos del CSV')
    st.write('Revisa la estructura de los datos limpios.')
    st.dataframe(df.head(30))

    st.markdown('---')

    # Extraer variables clave
    categoria_col = use_cols.get('categoria')
    valor_col = use_cols.get('valor')
    year_col = use_cols.get('year')

    st.header('Indicadores clave')
    
    # 1. Extraer el hito de mayores exportaciones en BPD
    mask_export_bpd = (df['unidades'].astype(str).str.lower().str.contains('bpd', na=False)) & (df[categoria_col].str.lower().str.contains('exportaci', na=False))
    export_bpd_df = df.loc[mask_export_bpd].dropna(subset=[valor_col])
    
    max_export_bpd = export_bpd_df[valor_col].max() if not export_bpd_df.empty else 0
    max_export_year = int(export_bpd_df.loc[export_bpd_df[valor_col].idxmax(), year_col]) if not export_bpd_df.empty else 'N/A'

    # 2. Extraer el hito financiero en USD
    mask_usd = df['unidades'].astype(str).str.lower().str.contains('usd', na=False)
    ingresos_usd = df.loc[mask_usd, valor_col].sum() if mask_usd.any() else 0

    # 3. Mostrar métricas organizadas
    col1, col2, col3 = st.columns(3)
    col1.metric('Pico Histórico Exportación (BPD)', f'{max_export_bpd:,.0f}')
    col2.metric('Año del Pico de Exportación', f'{max_export_year}')
    
    if ingresos_usd > 0:
        col3.metric('Ingresos Financieros Extraordinarios (USD)', f'${ingresos_usd:,.0f}')
    else:
        col3.metric('Total Registros Analizados', f'{len(df)}')

    st.markdown('---')

    st.header('Visualizaciones Históricas')
    st.write('Evolución en volumen (Barriles por Día).')

    figs = build_plots(df, use_cols, reservas_col, reservas_global_col)
    
    # Renderizamos la comparativa
    if 'line_produccion_export' in figs:
        st.pyplot(figs['line_produccion_export'])
    elif 'line_produccion' in figs:
        st.pyplot(figs['line_produccion'])
    
    st.markdown('---')
    st.subheader('Destinos de exportación')
    st.write('Frecuencia con la que un país/mercado es mencionado en los registros de exportación (BPD).')
    if 'bar_export_destinos' in figs:
        st.pyplot(figs['bar_export_destinos'])
    else:
        st.info('No hay suficientes datos de exportación para mostrar destinos.')

    st.markdown('---')
    st.subheader('Comparativa de reservas')
    st.write('Comparación entre reservas de Venezuela y las reservas globales disponibles en el último registro válido.')
    if 'pie_reservas' in figs:
        st.pyplot(figs['pie_reservas'])
    else:
        st.info('No hay datos completos de reservas para mostrar la comparativa.')


if __name__ == '__main__':
    main()