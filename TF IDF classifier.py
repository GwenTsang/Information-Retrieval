import argparse
import os
from pathlib import Path
import re
import json
import warnings
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss
from joblib import dump

def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return path.read_bytes().decode("utf-8", errors="replace")

    # Normalisation légère: plusieurs espace -> 1 espace, lignes -> espace (pour les fichiers issus de Gutenberg c'est nécessaire)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def collect_files(author_dir: Path, label: str):
    paths = sorted([p for p in author_dir.rglob("*.txt") if p.is_file()])
    X = []
    lengths = []
    for p in paths:
        txt = safe_read_text(p)
        X.append(txt)
        lengths.append(len(txt))
    y = [label] * len(paths)
    return paths, X, y, lengths


def build_dataset(balzac_dir: str, bergson_dir: str):
    balzac_dir = Path(balzac_dir)
    bergson_dir = Path(bergson_dir)

    b_paths, b_X, b_y, b_len = collect_files(balzac_dir, "Balzac")
    g_paths, g_X, g_y, g_len = collect_files(bergson_dir, "Bergson")

    file_paths = b_paths + g_paths
    texts = b_X + g_X
    labels = b_y + g_y
    lengths = b_len + g_len

    return file_paths, texts, labels, lengths

def make_pipeline(ngram_min=3, ngram_max=5, min_df=2, C=1.0, calibration="sigmoid", cv=3, max_features=None):
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
        max_features=max_features,
        dtype=np.float32,
    )

    base_clf = LinearSVC(C=C, class_weight="balanced")

    # Au cas où il y aurait une incompatibilité
    try:
        clf = CalibratedClassifierCV(estimator=base_clf, method=calibration, cv=cv)
    except TypeError:
        clf = CalibratedClassifierCV(base_estimator=base_clf, method=calibration, cv=cv)

    pipe = Pipeline([
        ("vect", vectorizer),
        ("clf", clf),
    ])
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Classifieur Bergson vs Balzac (fichiers .txt)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion d'entraînement (entre 0.7 et 0.9)")
    parser.add_argument("--ngram_min", type=int, default=3)
    parser.add_argument("--ngram_max", type=int, default=5)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--predict_unknown", type=str, default=None, help="Chemin d'un .txt inconnu à scorer après entraînement")
    args = parser.parse_args()

    # Paramètres fixes
    balzac_dir = "/content/Balzac"
    bergson_dir = "/content/Bergson"
    calibration = "sigmoid"
    output_dir = "/content/author_model"
    random_state = 42
    C = 1.0
    cv = 3
    max_features = None

    if not (0.7 <= args.train_ratio <= 0.9):
        print(f"[!] train_ratio {args.train_ratio} hors plage [0.7, 0.9]. Utilisation de 0.8.")
        args.train_ratio = 0.8

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Chargement des données (split au niveau fichier)
    file_paths, texts, labels, lengths = build_dataset(balzac_dir, bergson_dir)

    df_all = pl.DataFrame({
        "file_path": [str(p) for p in file_paths],
        "text": texts,
        "label": labels,
        "doc_length_chars": lengths,
    })

    # 2) Split stratifié (par fichier)
    X = df_all["text"].to_list()
    y = df_all["label"].to_list()
    paths = df_all["file_path"].to_list()
    lens = df_all["doc_length_chars"].to_list()

    X_train, X_test, y_train, y_test, paths_train, paths_test, lens_train, lens_test = train_test_split(
        X, y, paths, lens,
        train_size=args.train_ratio,
        stratify=y,
        random_state=random_state,
    )

    # 3) Retirer les fichiers vides de l'entraînement (mais les garder en test)
    non_empty_train_mask = [len(t) > 0 for t in X_train]
    if sum(non_empty_train_mask) == 0:
        raise RuntimeError("Tous les fichiers d'entraînement sont vides. Impossible d'entraîner un modèle.")
    if sum(non_empty_train_mask) < len(X_train):
        print(f"[i] {len(X_train) - sum(non_empty_train_mask)} fichier(s) vide(s) retiré(s) de l'entraînement.")

    X_train_ne = [t for t, m in zip(X_train, non_empty_train_mask) if m]
    y_train_ne = [l for l, m in zip(y_train, non_empty_train_mask) if m]

    # 4) Pipeline + entraînement
    pipe = make_pipeline(
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        C=C,
        calibration=calibration,
        cv=cv,
        max_features=max_features
    )

    print("[i] Entraînement")
    pipe.fit(X_train_ne, y_train_ne)

    # 5) Évaluation + probabilités
    print("[i] Évaluation sur le test")
    y_pred = pipe.predict(X_test)

    # Probabilités
    if hasattr(pipe.named_steps["clf"], "classes_"):
        classes = list(pipe.named_steps["clf"].classes_)
    else:
        # Repli (rare): tenter sur le pipeline directement
        classes = list(pipe.classes_)

    proba = pipe.predict_proba(X_test)  # shape: (n_samples, 2)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_bergson = class_to_idx.get("Bergson")
    idx_balzac = class_to_idx.get("Balzac")

    prob_bergson = proba[:, idx_bergson] if idx_bergson is not None else np.nan
    prob_balzac = proba[:, idx_balzac] if idx_balzac is not None else np.nan

    acc = accuracy_score(y_test, y_pred)
    try:
        ll = log_loss(y_test, proba, labels=classes)
    except Exception:
        ll = np.nan

    print(f"[i] Accuracy test: {acc:.4f} | Log-loss: {ll if isinstance(ll, float) else 'N/A'}")

    # 6) Export CSV des prédictions (test uniquement)
    df_pred = pl.DataFrame({
        "file_path": paths_test,
        "true_label": y_test,
        "pred_label": y_pred,
        "prob_Bergson": prob_bergson,
        "prob_Balzac": prob_balzac,
        "doc_length_chars": lens_test,
    })

    csv_path = out_dir / "predictions.csv"
    df_pred.write_csv(csv_path)
    print(f"[i] predictions.csv écrit: {csv_path}")

    # 7) Sauvegarde du modèle et des artefacts
    model_path = out_dir / "author_classifier.joblib"
    dump(pipe, model_path)
    print(f"[i] Modèle sauvegardé: {model_path}")

    # Sauvegarde d'un petit manifest
    manifest = {
        "train_ratio": args.train_ratio,
        "random_state": random_state,
        "vectorizer": {
            "analyzer": "char",
            "ngram_range": [args.ngram_min, args.ngram_max],
            "min_df": args.min_df,
            "max_features": max_features,
            "strip_accents": "unicode",
            "lowercase": True,
            "sublinear_tf": True
        },
        "classifier": "LinearSVC + CalibratedClassifierCV",
        "C": C,
        "calibration": calibration,
        "cv": cv,
        "classes": classes,
        "metrics": {"test_accuracy": acc, "test_log_loss": ll},
        "n_train_files": len(X_train),
        "n_test_files": len(X_test)
    }
    with open(out_dir / "model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 8) Scorer un fichier inconnu après entraînement
    if args.predict_unknown:
        upath = Path(args.predict_unknown)
        if not upath.exists():
            print(f"[!] Fichier inconnu introuvable: {upath}")
        else:
            utext = safe_read_text(upath)
            u_proba = pipe.predict_proba([utext])[0]
            u_pred = pipe.predict([utext])[0]
            p_bergson = u_proba[class_to_idx.get("Bergson")] if class_to_idx.get("Bergson") is not None else np.nan
            p_balzac = u_proba[class_to_idx.get("Balzac")] if class_to_idx.get("Balzac") is not None else np.nan
            print(f"[i] Fichier: {upath}")
            print(f"    Prédiction: {u_pred}")
            print(f"    P(Bergson) {p_bergson:.4f} | P(Balzac) {p_balzac:.4f}")


if __name__ == "__main__":
    main()