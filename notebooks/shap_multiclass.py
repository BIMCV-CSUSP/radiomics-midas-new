###########################
#         SHAP            #
###########################

def perform_shap_analysis(X_data, y_data, model_clf, preprocessor, shap_dir, report_path, dataset_name="conjunto"):
    """
    Realiza análisis SHAP sobre un conjunto de datos.
    
    Args:
        X_data: Datos de características sin procesar
        y_data: Etiquetas
        model_clf: Clasificador final
        preprocessor: Pipeline de preprocesamiento
        shap_dir: Directorio donde guardar los resultados
        dataset_name: Nombre del conjunto de datos (para etiquetar)
    """
    print(f"\nRealizando análisis SHAP para {dataset_name}...")
    try:
        print(X_data.shape)
        # Aplicar StandardScaler conservando nombres de columnas
        scaler = preprocessor.steps[0][1]
        X_scaled = pd.DataFrame(scaler.transform(X_data),
                            index=X_data.index,
                            columns=X_data.columns)
        # Aplicar VarianceThreshold y recuperar columnas seleccionadas
        vt = preprocessor.steps[1][1]
        mask = vt.get_support()
        selected_features = X_data.columns[mask]
        X_transformed_array = vt.transform(X_scaled.values)
        X_transformed = pd.DataFrame(X_transformed_array,
                                    index=X_data.index,
                                    columns=selected_features)
        # # Seleccionar el explainer adecuado según el tipo de modelo
        # if isinstance(model_clf, (RandomForestClassifier, GradientBoostingClassifier)):
        #     # Para modelos basados en árboles
        #     explainer = shap.TreeExplainer(model_clf)
        # elif isinstance(model_clf, LogisticRegression):
        #     # Para modelos lineales
        #     try:
        #         explainer = shap.LinearExplainer(model_clf, X_transformed)
        #     except Exception:
        #         # Si falla, usar KernelExplainer como alternativa
        #         background = shap.kmeans(X_transformed, 50)
        #         explainer = shap.KernelExplainer(model_clf.predict_proba, background)
        # else:
        #     # Para otros modelos (SVM, KNN, NaiveBayes)
        #     background = shap.kmeans(X_transformed, 50) # Resumen del dataset para acelerar
        #     explainer = shap.KernelExplainer(model_clf.predict_proba, background)
        
        #prueba
        print(" - Preprocesando datos...")
        classes = np.unique(y_data)
        class_names = {i: f"Clase {c}" for i,c in enumerate(classes)}
        print(f" - Clases detectadas: {class_names}")
        import os, joblib

        shap_cache = os.path.join(shap_dir, "shap_background_cache22.pkl")

        if os.path.exists(shap_cache):
            # load previously‐computed explainer result
            shap_result = joblib.load(shap_cache)
            X_slice = X_transformed #shap_result.data
        else:
            # pick a small background to speed things up
            background = X_transformed.sample(min(200, len(X_transformed)), random_state=42)

            # build the explainer on predict_proba
            explainer = shap.Explainer(
                model_clf.predict_proba,      
                masker=Independent(background),
                feature_names=selected_features,
                output_names=class_names       
            )

            # compute SHAP values (on a slice or on the whole)
            X_slice = X_transformed 
            shap_result = explainer(X_slice)    

            # cache it
            joblib.dump(shap_result, shap_cache)

        # now you can immediately extract and plot:
        all_vals = shap_result.values
        print(f" - Valores SHAP calculados para {dataset_name}.")
        # Verificar si es un problema de clasificación multiclase
        if len(all_vals.shape) == 2:
            # Caso binario o regresión
            shap_values_class = [all_vals]
        elif len(all_vals.shape) == 3:
            # Caso multiclase
            # all_vals tiene forma (n_samples, n_features, n_classes)
            print(f" - Detectado problema de clasificación multiclase con {all_vals.shape[2]} clases.")
            # Extraer valores SHAP por clase
            if all_vals.shape[0] != X_slice.shape[0]:
                raise ValueError("Los valores SHAP no coinciden con el número de muestras en X_slice.")
            if all_vals.shape[1] != X_slice.shape[1]:
                raise ValueError("Los valores SHAP no coinciden con el número de características en X_slice.")
        n_classes = all_vals.shape[2]
        shap_values_class = [all_vals[:, :, i] for i in range(n_classes)]

        shap.summary_plot(
            all_vals,
            X_slice,           # or X_transformed for full
            feature_names=selected_features,
            class_names=class_names,
            class_inds="original",
            show=False
        )
        plt.savefig(os.path.join(shap_dir, "shap_summary_multiclass.png"),
                    bbox_inches="tight", dpi=300)
        plt.close()
        
        # --------------------------------------------------------------
        # PARTE 1: TEST ESTADÍSTICO entre valores SHAP y la clase
        # --------------------------------------------------------------
        print(f" - Realizando test estadístico (Kruskal-Wallis) para {dataset_name} con corrección Holm...")

        # Detectar clases únicas
        unique_classes = np.unique(y_data)
        # Lista para guardar resultados por clase
        shap_stats_results = []

        # Procesar una matriz SHAP por cada clase
        for class_idx, class_shap_values in enumerate(shap_values_class):
            print(f"  > Procesando SHAP para la clase {class_idx}...")

            shap_matrix = pd.DataFrame(
                class_shap_values,
                index=X_transformed.index,
                columns=X_transformed.columns
            )

            features_test = []
            pvalues_raw = []

            for feat in shap_matrix.columns:
                # Agrupar valores SHAP de esta feature por clase
                grouped_values = [shap_matrix.loc[y_data == c, feat] for c in unique_classes]
                try:
                    stat, pval = kruskal(*grouped_values)
                except ValueError:
                    pval = 1.0  # fallback in case one class has no samples
                features_test.append(feat)
                pvalues_raw.append(pval)

            # Corrección Holm por comparaciones múltiples
            alpha = 0.05
            reject, pvals_corr, _, _ = multipletests(pvalues_raw, alpha=alpha, method='holm')

            # Preparar el reporte
            lines_output = []
            lines_output.append("=================================")
            lines_output.append(f"Kruskal-Wallis por feature - SHAP clase {class_idx}")
            lines_output.append(f"alpha = {alpha}")
            lines_output.append(f"Features totales: {len(features_test)}") 
            lines_output.append("=================================\n")
            lines_output.append(f"Resultados por feature (p-valor crudo y corregido):")

            significant_feats = []

            for feat, pval_raw, pval_corr, rej_bool in zip(features_test, pvalues_raw, pvals_corr, reject):
                if rej_bool:
                    result_str = "=> DIFERENCIA SIGNIFICATIVA"
                    significant_feats.append((feat, pval_raw, pval_corr))
                else:
                    result_str = "=> sin diferencia significativa"
                lines_output.append(
                    f"    {feat}: p-valor crudo={pval_raw:.4e}, p-valor corregido={pval_corr:.4e} {result_str}"
                )

            lines_output.append("")
            lines_output.append(f" Total con diferencia significativa: {len(significant_feats)}")

            shap_stats_results.append((class_idx, lines_output))

            # Guardar resultado para esta clase
            test_txt_path = os.path.join(shap_dir, f"shap_statistical_test_class_{class_idx}.txt")
            with open(test_txt_path, "w", encoding="utf-8") as f_out:
                for line in lines_output:
                    f_out.write(line + "\n")

            print(f"    --> Guardado: {test_txt_path}")

    
        # --------------------------------------------------------------
        # PARTE 2: HEATMAP
        # --------------------------------------------------------------
        print(f" - Generando Heatmap con muestras ordenadas por clase para {dataset_name}...")
        # Obtener índices ordenados por clase
        class_labels = np.unique(y_data)
        idx_order = np.concatenate([np.where(y_data == c)[0] for c in class_labels])
        class_positions = np.cumsum([len(np.where(y_data == c)[0]) for c in class_labels])

        for i, class_shap in enumerate(shap_values_class):
            print(f" - Generando heatmap para Clase {i}...")
            heatmap_path = os.path.join(shap_dir, f"shap_heatmap_class_{i}.png")
            
            shap.plots.heatmap(
                class_shap,
                show=False,
                instance_order=idx_order
            )
            fig = plt.gcf()
            ax = plt.gca()
            
            # Dibujar líneas divisorias entre clases
            for split in class_positions[:-1]:
                ax.axvline(split - 0.5, color='black', linewidth=1, zorder=10)

            # Etiquetas de clase
            n_total = len(idx_order)
            prev = 0
            for c, split in zip(class_labels, class_positions):
                midpoint = (prev + split) / 2 / n_total
                ax.text(midpoint, 1.01, f'Clase {c}', ha='center', va='bottom', transform=ax.transAxes)
                prev = split

            fig.set_size_inches(10, 6)
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  --> Heatmap guardado: {heatmap_path}")

    
        # --------------
        # Beeswarm plot 
        # --------------
        for i, class_shap in enumerate(shap_values_class):
            shap_fig_path = os.path.join(shap_dir, f"shap_beeswarm_class_{i}.png")
            shap.plots.beeswarm(class_shap, max_display=16, show=False)
            fig = plt.gcf()
            fig.set_size_inches(14, 8)
            plt.subplots_adjust(left=0.4, right=0.95)
            plt.tight_layout()
            plt.savefig(shap_fig_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            print(f"  --> Beeswarm plot guardado: {shap_fig_path}")

    
        # --------------------------------------------------------------
        # Scatter plots de las top features
        # --------------------------------------------------------------
        # Crear directorio para gráficos individuales
        top_features_by_class = []
        for i, class_shap in enumerate(shap_values_class):
            mean_abs_shap = np.abs(class_shap.values).mean(axis=0)
            top_idx = np.argsort(mean_abs_shap)[-15:]
            top_idx = top_idx[np.argsort(mean_abs_shap[top_idx])[::-1]]
            top_features_shap = X_transformed.columns[top_idx]
            top_features_by_class.append(top_features_shap)
            
            scatter_dir_class = os.path.join(shap_dir, f"scatter_plots_class_{i}")
            os.makedirs(scatter_dir_class, exist_ok=True)

            for j, feature in enumerate(top_features_shap, start=1):
                scatter_fig_path = os.path.join(scatter_dir_class, f"{j:02d}_{feature}.png")
                shap.plots.scatter(class_shap[:, feature], color=class_shap, show=False)
                fig = plt.gcf()
                fig.set_size_inches(10, 6)
                plt.tight_layout()
                plt.savefig(scatter_fig_path, dpi=dpi, bbox_inches='tight')
                plt.close()
            
            print(f"  --> Scatter plots guardados para clase {i} en: {scatter_dir_class}")

        return True, selected_features, shap_values_class, top_features_shap
    
    except Exception as e:
        with open(report_path, "a", encoding="utf-8") as f_out:
            f_out.write(f"=== SHAP Analysis ({dataset_name}) ===\n")
            f_out.write(" No se pudo generar SHAP (modelo no soportado o error):\n")
            f_out.write(f"  {repr(e)}\n\n")
        print(f"Error en SHAP analysis para {dataset_name}:", e)
        return False, None, None, None
    
    
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from copy import deepcopy
from pathlib import Path
import joblib
from sklearn.metrics import make_scorer, roc_auc_score


import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from shap.maskers import Independent


# Paths and settings
path_features        = '/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/binary2/features_t2w_MPfirrmann.csv'
experiment_dir       = '/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/data/features_t2w_multiclass'
variables_txt_path   = os.path.join(experiment_dir, 'variables_usadas.txt')
best_model_finetune  = 'GradientBoosting'
n_folds              = 5

# Load data
df = pd.read_csv(path_features)
print("Configuration:")
print(f"  Features CSV:       {path_features}")
print(f"  Experiment dir:     {experiment_dir}")
print(f"  Variables file:     {variables_txt_path}")
print(f"  Model for fine-tune:{best_model_finetune}")
print(f"  CV folds:           {n_folds}")
print(f"\nLoaded DataFrame with shape {df.shape}")



# Configuración de rutas y directorios de salida
selected_model = best_model_finetune
base_dir = os.path.dirname(os.path.abspath(variables_txt_path))
output_parent_dir = os.path.join(base_dir, f"best_results")
calibration_dir = os.path.join(output_parent_dir, "calibration")
explicability_dir = os.path.join(output_parent_dir, "explicability")

train_explicability_dir = os.path.join(explicability_dir, "train")
test_explicability_dir = os.path.join(explicability_dir, "test")

# Subdirectorios SHAP y LIME para train
train_shap_dir = os.path.join(train_explicability_dir, "SHAP")
train_lime_dir = os.path.join(train_explicability_dir, "LIME")

# Subdirectorios SHAP y LIME para test
test_shap_dir = os.path.join(test_explicability_dir, "SHAP")
test_lime_dir = os.path.join(test_explicability_dir, "LIME")

# Crear directorios si no existen
os.makedirs(output_parent_dir, exist_ok=True)
os.makedirs(calibration_dir, exist_ok=True)
os.makedirs(explicability_dir, exist_ok=True)
os.makedirs(train_explicability_dir, exist_ok=True)
os.makedirs(test_explicability_dir, exist_ok=True)
os.makedirs(train_shap_dir, exist_ok=True)
os.makedirs(train_lime_dir, exist_ok=True)
os.makedirs(test_shap_dir, exist_ok=True)
os.makedirs(test_lime_dir, exist_ok=True)

print(f"\nCarpeta de salida creada/ubicada en: {os.path.relpath(output_parent_dir)}")



# ----------------------------------------------------------------------
# 1) CARGAR CSV E IDENTIFICAR X, y, groups
# ----------------------------------------------------------------------
df = pd.read_csv(path_features)

df['patient_id_type'] = df['patient_id'].astype(str)
df = df.set_index('patient_id_type')
print(f"Datos cargados. Dimensiones: {df.shape}")

# Preparar variables para el modelado
y = df['label'].values
groups = df['patient_id'].values
X = df.drop(columns=['patient_id'])

# ----------------------------------------------------------------------
# 1.1) FILTRAR LAS VARIABLES USADAS (variables_usadas.txt)
# ----------------------------------------------------------------------
print(f"\nFiltrando variables usando el archivo: {variables_txt_path}")
with open(variables_txt_path, "r", encoding="utf-8") as f_vars:
    used_vars = [line.strip() for line in f_vars if line.strip()]
X = X[used_vars]

# ----------------------------------------------------------------------
# 2) SEPARAR HOLD-OUT TEST SET Y CONJUNTO DE ENTRENAMIENTO
# ----------------------------------------------------------------------
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train_full, y_test = y[train_idx], y[test_idx]
groups_train_full = groups[train_idx]

#Split data for small debugging
# max_debug = 200
# n = len(X_train_full)
# if n > max_debug:
#     # pick 200 random integer positions
#     sel = np.random.RandomState(42).choice(n, size=max_debug, replace=False)
#     X_data = X_train_full.iloc[sel]
#     y_data = y_train_full[sel]
#     groups_train_full = groups_train_full[sel]
X_data = X_train_full
y_data = y_train_full


roc_auc_ovr = make_scorer(
    roc_auc_score,
    response_method="predict_proba",
    multi_class="ovr",
    average="macro"
)

score_group = {
    "roc_auc_ovr": roc_auc_ovr,
    "f1":            "f1_macro",
    "balanced_accuracy": "balanced_accuracy"
}
score_refit_str = "roc_auc_ovr"
random_state_value = 42 


output_parent_dir = Path("../data/features_t2w_multiclass/best_results")
estimator_path     = output_parent_dir / "best_estimator.pkl"
# load the fitted pipeline
best_estimator = joblib.load(estimator_path)
# Extraer el preprocesador (todos los pasos excepto el clasificador final)
preprocessor = deepcopy(best_estimator)
preprocessor.steps.pop(-1)

# Extraer el clasificador final
model_clf = best_estimator.steps[-1][1]
# pull out only the GB parameters
gb_params = {
    k: v
    for k, v in best_estimator.get_params().items()
    if k.startswith("gradientboostingclassifier__")
}

print("Optimización completada.")
print(" → Mejores parámetros:", gb_params)
print(" → Mejor estimador cargado desde:", estimator_path)

report_path = os.path.join(output_parent_dir, "report.txt")
# with open(report_path, "w", encoding="utf-8") as f_out:
#     f_out.write(f"=== Fine-tuning del modelo {selected_model} ===\n\n")
#     f_out.write(f"Mejores parámetros (según {score_refit_str}): {search.best_params_}\n\n")
#     f_out.write("=== Resultados CV (BayesSearch) ===\n")
#     idx_best = search.best_index_
#     for key in score_group:
#         mean_test = search.cv_results_[f'mean_test_{key}'][idx_best]
#         std_test  = search.cv_results_[f'std_test_{key}'][idx_best]
#         f_out.write(f"  CV {key}: {mean_test:.3f} +/- {std_test:.3f}\n")
#     f_out.write("\n")


import joblib
import os

shap_cache_file = os.path.join(train_shap_dir, "shap_analysis_cache_new2.pkl")

if os.path.exists(shap_cache_file):
    print("→ Loading cached SHAP analysis results...")
    train_success, selected_features, train_shap_values, train_top_features = joblib.load(shap_cache_file)
else:
    print("→ Running SHAP analysis...")
    train_success, selected_features, train_shap_values, train_top_features = perform_shap_analysis(
        X_data=X_train_full,
        y_data=y_train_full,
        model_clf=model_clf,
        preprocessor=preprocessor,
        shap_dir=train_shap_dir,
        report_path=report_path,
        dataset_name="entrenamiento"
    )
    joblib.dump((train_success, selected_features, train_shap_values, train_top_features), shap_cache_file)
    print(f"→ SHAP analysis results cached in {shap_cache_file}")
