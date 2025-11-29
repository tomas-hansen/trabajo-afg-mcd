# Integración de Modelos en el Optimizador

## Resumen
Se ha integrado todos los modelos entrenados en el notebook "Entrenamiento definitivo modelos CBM.ipynb" en el optimizador (`optimizer.py`), permitiendo seleccionar entre diferentes motores de predicción para optimizar los reactivos de flotación.

## Modelos Disponibles
- **LSTM Simple**: Solo usa lags del target (% Iron Concentrate).
- **LSTM Full**: Usa todas las variables disponibles.
- **LightGBM**: Modelo de árbol optimizado con Optuna (sin lags del target).
- **LightGBM Full**: Modelo de árbol multivariable con todas las variables.
- **PLS**: Partial Least Squares como baseline estático.
- **XGBoost**: Modelo existente (si disponible).

## Archivos Modificados
- `train_models.py`: Script para entrenar y empaquetar todos los modelos en bundles (zip).
- `utils.py`: Agregadas funciones `load_lgbm_bundle` y `load_pls_bundle`, y `make_objective_sklearn` para optimización con modelos sklearn.
- `optimizer.py`: Modificado para cargar múltiples bundles y seleccionar engine.

## Uso
1. **Entrenar Modelos**: Ejecutar `python train_models.py` para entrenar y crear bundles (lstm_simple_bundle.zip, lstm_full_bundle.zip, etc.).
2. **Evaluar Modelos**: Ejecutar `python evaluate_models.py` para ver el rendimiento de todos los modelos en el conjunto de test (sin reentrenar).
3. **Probar Optimizador**: Ejecutar `python test_optimizer.py` para probar la optimización con todos los engines disponibles.
4. **Optimización Individual**: Ejecutar `python optimizer.py` con parámetro engine:
   - `engine="auto"`: Selecciona automáticamente el mejor disponible (prioriza LSTM full).
   - `engine="lstm_simple"`: Usa LSTM simple.
   - `engine="lstm_full"`: Usa LSTM full.
   - `engine="xgb"`: Usa XGBoost.
   - `engine="lgbm"`: Usa LightGBM.
   - `engine="pls"`: Usa PLS.
   - `engine="lgbm_full"`: Usa LightGBM full.

## Ejemplo
```bash
python optimizer.py  # Usa auto
```

O modificar el main:
```python
main(engine="pls")
```

## Notas
- Los bundles se guardan en directorios separados y se zipean automáticamente.
- Si un bundle no existe, se omite con warning.
- La optimización usa evolución diferencial para maximizar la predicción de hierro concentrado.
- Se reporta mejora validada con XGBoost si disponible.