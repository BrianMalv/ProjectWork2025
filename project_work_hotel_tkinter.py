
import os
import random
import string
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import tkinter as tk
from tkinter import filedialog, messagebox


# =========================
# 1. GENERAZIONE DATASET
# =========================

def generate_synthetic_reviews(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:


    random.seed(random_state)
    np.random.seed(random_state)

    departments = ["Housekeeping", "Reception", "F&B"]
    sentiments = ["pos", "neg"]

    templates = {
        "Housekeeping": {
            "pos_titles": [
                "Camera pulitissima",
                "Ottimo servizio di pulizia",
                "Stanza impeccabile",
                "Pulizia eccellente",
            ],
            "pos_bodies": [
                "La camera era sempre pulita e profumata, asciugamani cambiati ogni giorno.",
                "Servizio di pulizia rapido e accurato, letto rifatto alla perfezione.",
                "Bagno brillante e senza tracce di sporco, complimenti alle cameriere ai piani.",
                "Pulizia della stanza davvero ottima, non ho trovato un granello di polvere.",
            ],
            "neg_titles": [
                "Camera sporca",
                "Pulizia insufficiente",
                "Stanza trascurata",
                "Deluso dalla pulizia",
            ],
            "neg_bodies": [
                "Ho trovato polvere sui mobili e capelli nel bagno.",
                "La camera non è stata rifatta un giorno su due, davvero inaccettabile.",
                "Macchie sulle lenzuola e asciugamani non cambiati.",
                "Pulizia molto approssimativa, pavimento appiccicoso e cestino non svuotato.",
            ],
        },
        "Reception": {
            "pos_titles": [
                "Accoglienza perfetta",
                "Staff gentilissimo",
                "Check-in veloce",
                "Reception cordiale",
            ],
            "pos_bodies": [
                "Personale della reception molto disponibile e sorridente.",
                "Check-in rapido e senza problemi, ci hanno dato anche ottimi consigli sulla città.",
                "Ottima gestione del check-out, conto chiaro e nessun intoppo.",
                "Reception aperta 24 ore su 24, sempre pronti ad aiutare.",
            ],
            "neg_titles": [
                "Check-in lunghissimo",
                "Reception poco disponibile",
                "Problemi con il pagamento",
                "Esperienza negativa al desk",
            ],
            "neg_bodies": [
                "Ho dovuto aspettare più di mezz'ora per il check-in.",
                "Personale poco cordiale, nessun sorriso e poca attenzione alle richieste.",
                "Hanno sbagliato a registrare il pagamento e ho dovuto discutere per sistemare.",
                "Informazioni confuse su tasse di soggiorno e orari, poca professionalità.",
            ],
        },
        "F&B": {
            "pos_titles": [
                "Colazione fantastica",
                "Ristorante eccellente",
                "Ottimo buffet",
                "Cibo delizioso",
            ],
            "pos_bodies": [
                "Colazione abbondante con molta scelta di dolce e salato.",
                "Ristorante dell'hotel davvero buono, piatti curati e servizio puntuale.",
                "Buffet serale molto vario, qualità del cibo sopra la media.",
                "Camerieri gentili e servizio veloce, cibo di ottima qualità.",
            ],
            "neg_titles": [
                "Colazione scarsa",
                "Ristorante deludente",
                "Cibo freddo",
                "Esperienza pessima al buffet",
            ],
            "neg_bodies": [
                "Poca scelta a colazione, molti prodotti confezionati.",
                "Il ristorante era caro e la qualità dei piatti non all'altezza.",
                "Cibo servito tiepido e poco saporito.",
                "Buffet quasi vuoto e piatti non riforniti, molto deludente.",
            ],
        },
    }

    ambiguous_bodies = [
        "Camera pulita ma colazione deludente, non so se tornerei.",
        "Check-in veloce ma la stanza aveva qualche problema.",
        "Personale gentile alla reception, però il ristorante non mi ha convinto.",
        "Colazione buona ma il bagno non era perfetto.",
    ]

    rows = []

    for i in range(1, n_samples + 1):
        dept = random.choice(departments)
        sentiment = random.choice(sentiments)

        dept_templates = templates[dept]

        if sentiment == "pos":
            title = random.choice(dept_templates["pos_titles"])
            body = random.choice(dept_templates["pos_bodies"])
        else:
            title = random.choice(dept_templates["neg_titles"])
            body = random.choice(dept_templates["neg_bodies"])

        extra = ""
        if random.random() < 0.15:
            extra = " " + random.choice(ambiguous_bodies)

        full_body = body + extra

        rows.append(
            {
                "id": i,
                "title": title,
                "body": full_body,
                "department": dept,
                "sentiment": sentiment,
            }
        )

    df = pd.DataFrame(rows)
    return df


# =========================
# 2. PIPELINE ML
# =========================

def simple_preprocess(text: str) -> str:
    """
    Preprocessing minimale:
    - minuscole
    - rimozione punteggiatura
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def plot_confusion_matrix(cm, class_names, title, filename):
    """
    Salva una confusion matrix come immagine PNG.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def train_models(df: pd.DataFrame):
    """
    Addestra i modelli per:
    - department (multiclasse)
    - sentiment (binario)
    Restituisce: vectorizer, dept_model, sent_model, metrics_dict
    """
    df["text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).apply(
        simple_preprocess
    )

    X_text = df["text"].values
    y_dept = df["department"].values
    y_sent = df["sentiment"].values

    X_train, X_test, y_dept_train, y_dept_test, y_sent_train, y_sent_test = train_test_split(
        X_text, y_dept, y_sent, test_size=0.2, random_state=42, stratify=y_dept
    )

    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Reparto
    dept_clf = LogisticRegression(max_iter=200, multi_class="auto")
    dept_clf.fit(X_train_vec, y_dept_train)

    # Sentiment
    sent_clf = LogisticRegression(max_iter=200)
    sent_clf.fit(X_train_vec, y_sent_train)

    # Valutazione reparto
    y_dept_pred = dept_clf.predict(X_test_vec)
    dept_acc = accuracy_score(y_dept_test, y_dept_pred)
    dept_f1_macro = f1_score(y_dept_test, y_dept_pred, average="macro")
    dept_report = classification_report(y_dept_test, y_dept_pred)

    dept_classes = np.unique(y_dept)
    cm_dept = confusion_matrix(y_dept_test, y_dept_pred, labels=dept_classes)
    os.makedirs("results", exist_ok=True)
    plot_confusion_matrix(
        cm_dept,
        class_names=dept_classes,
        title="Confusion Matrix - Reparto",
        filename=os.path.join("results", "confusion_matrix_department.png"),
    )

    # Valutazione sentiment
    y_sent_pred = sent_clf.predict(X_test_vec)
    sent_acc = accuracy_score(y_sent_test, y_sent_pred)
    sent_f1_macro = f1_score(y_sent_test, y_sent_pred, average="macro")
    sent_report = classification_report(y_sent_test, y_sent_pred)

    sent_classes = np.unique(y_sent)
    cm_sent = confusion_matrix(y_sent_test, y_sent_pred, labels=sent_classes)
    plot_confusion_matrix(
        cm_sent,
        class_names=sent_classes,
        title="Confusion Matrix - Sentiment",
        filename=os.path.join("results", "confusion_matrix_sentiment.png"),
    )

    print("\n=== RISULTATI REPARTO ===")
    print(f"Accuracy reparto: {dept_acc:.3f}")
    print(f"F1 macro reparto: {dept_f1_macro:.3f}")
    print(dept_report)

    print("\n=== RISULTATI SENTIMENT ===")
    print(f"Accuracy sentiment: {sent_acc:.3f}")
    print(f"F1 macro sentiment: {sent_f1_macro:.3f}")
    print(sent_report)

    metrics = {
        "dept_acc": dept_acc,
        "dept_f1": dept_f1_macro,
        "sent_acc": sent_acc,
        "sent_f1": sent_f1_macro,
    }

    # Esempi di errori
    print("\n=== ESEMPI DI ERRORI (max 5) ===")
    X_test_list = list(X_test)
    count_errors = 0
    for i, text in enumerate(X_test_list):
        wrong_dept = y_dept_test[i] != y_dept_pred[i]
        wrong_sent = y_sent_test[i] != y_sent_pred[i]
        if wrong_dept or wrong_sent:
            print("-" * 80)
            print(f"Testo: {text}")
            print(f"Reparto vero: {y_dept_test[i]} | Reparto predetto: {y_dept_pred[i]}")
            print(f"Sentiment vero: {y_sent_test[i]} | Sentiment predetto: {y_sent_pred[i]}")
            count_errors += 1
        if count_errors >= 5:
            break

    return vectorizer, dept_clf, sent_clf, metrics


# =========================
# 3. DASHBOARD TKINTER
# =========================

class ReviewDashboard(tk.Tk):
    def __init__(self, vectorizer, dept_model, sent_model, metrics):
        super().__init__()

        self.title("Hotel Reviews Routing & Sentiment - Project Work")
        self.geometry("950x700")

        self.vectorizer = vectorizer
        self.dept_model = dept_model
        self.sent_model = sent_model
        self.metrics = metrics

        self._build_ui()

    def _build_ui(self):
        # Sezione metriche
        frame_metrics = tk.LabelFrame(self, text="Metriche modello (test set 20%)")
        frame_metrics.pack(fill="x", padx=10, pady=10)

        lbl_metrics = tk.Label(
            frame_metrics,
            text=(
                f"Reparto -> Accuracy: {self.metrics['dept_acc']:.3f} | "
                f"F1 macro: {self.metrics['dept_f1']:.3f}    |    "
                f"Sentiment -> Accuracy: {self.metrics['sent_acc']:.3f} | "
                f"F1 macro: {self.metrics['sent_f1']:.3f}"
            ),
        )
        lbl_metrics.pack(anchor="w", padx=10, pady=5)

        # Titolo recensione
        lbl_title = tk.Label(self, text="Titolo recensione:")
        lbl_title.pack(anchor="w", padx=10, pady=(10, 0))

        self.entry_title = tk.Entry(self, width=120)
        self.entry_title.pack(anchor="w", padx=10, pady=5)

        # Testo recensione
        lbl_body = tk.Label(self, text="Testo recensione:")
        lbl_body.pack(anchor="w", padx=10, pady=(10, 0))

        self.text_body = tk.Text(self, width=120, height=10)
        self.text_body.pack(anchor="w", padx=10, pady=5)

        # Bottoni
        frame_buttons = tk.Frame(self)
        frame_buttons.pack(anchor="w", padx=10, pady=10)

        btn_predict = tk.Button(frame_buttons, text="Predici singola recensione", command=self.predict_single)
        btn_predict.grid(row=0, column=0, padx=5)

        btn_batch = tk.Button(frame_buttons, text="Predici da CSV", command=self.predict_from_csv)
        btn_batch.grid(row=0, column=1, padx=5)

        # Output
        lbl_output = tk.Label(self, text="Output:")
        lbl_output.pack(anchor="w", padx=10, pady=(10, 0))

        self.text_output = tk.Text(self, width=120, height=14, state="disabled")
        self.text_output.pack(anchor="w", padx=10, pady=5)

    def predict_single(self):
        title = self.entry_title.get().strip()
        body = self.text_body.get("1.0", tk.END).strip()

        if not title and not body:
            messagebox.showwarning("Attenzione", "Inserisci almeno il titolo o il testo della recensione.")
            return

        combined = simple_preprocess(title + " " + body)
        X_vec = self.vectorizer.transform([combined])

        # Reparto
        dept_probs = self.dept_model.predict_proba(X_vec)[0]
        dept_classes = self.dept_model.classes_
        dept_idx = dept_probs.argmax()
        dept_pred = dept_classes[dept_idx]
        dept_prob = dept_probs[dept_idx]

        # Sentiment
        sent_probs = self.sent_model.predict_proba(X_vec)[0]
        sent_classes = self.sent_model.classes_
        sent_idx = sent_probs.argmax()
        sent_pred = sent_classes[sent_idx]
        sent_prob = sent_probs[sent_idx]

        sent_label_it = "Positivo" if sent_pred == "pos" else "Negativo"

        lines = []
        lines.append(f"Reparto consigliato: {dept_pred} (probabilità {dept_prob:.2%})")
        lines.append(f"Sentiment stimato: {sent_label_it} [{sent_pred}] (probabilità {sent_prob:.2%})")
        lines.append("")
        lines.append("Probabilità reparto:")
        for cls, p in zip(dept_classes, dept_probs):
            lines.append(f" - {cls}: {p:.2%}")
        lines.append("")
        lines.append("Probabilità sentiment:")
        for cls, p in zip(sent_classes, sent_probs):
            label_it = "Positivo" if cls == "pos" else "Negativo"
            lines.append(f" - {label_it} [{cls}]: {p:.2%}")

        self.text_output.config(state="normal")
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert(tk.END, "\n".join(lines))
        self.text_output.config(state="disabled")

    def predict_from_csv(self):
        file_path = filedialog.askopenfilename(
            title="Seleziona CSV di recensioni",
            filetypes=[("File CSV", "*.csv")],
        )
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile leggere il file CSV: {e}")
            return

        required_cols = {"title", "body"}
        if not required_cols.issubset(set(df.columns)):
            messagebox.showerror(
                "Errore",
                f"Il CSV deve contenere almeno le colonne: {required_cols}",
            )
            return

        df["text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).apply(simple_preprocess)
        X_vec = self.vectorizer.transform(df["text"].values)

        # Reparto
        dept_probs = self.dept_model.predict_proba(X_vec)
        dept_classes = self.dept_model.classes_
        dept_pred_idx = dept_probs.argmax(axis=1)
        dept_pred = dept_classes[dept_pred_idx]
        dept_pred_prob = dept_probs.max(axis=1)

        # Sentiment
        sent_probs = self.sent_model.predict_proba(X_vec)
        sent_classes = self.sent_model.classes_
        sent_pred_idx = sent_probs.argmax(axis=1)
        sent_pred = sent_classes[sent_pred_idx]
        sent_pred_prob = sent_probs.max(axis=1)

        df["pred_department"] = dept_pred
        df["pred_department_prob"] = dept_pred_prob
        df["pred_sentiment"] = sent_pred
        df["pred_sentiment_prob"] = sent_pred_prob

        base, ext = os.path.splitext(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"{base}_predictions_{timestamp}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")

        messagebox.showinfo(
            "Completato",
            f"Predizioni salvate in:\n{out_path}",
        )


# =========================
# 4. MAIN
# =========================

def main():
    # Genera dataset sintetico
    df = generate_synthetic_reviews(n_samples=500, random_state=42)
    df.to_csv("reviews_synthetic.csv", index=False, encoding="utf-8")
    print("[INFO] Dataset sintetico salvato in reviews_synthetic.csv")
    print(df.head())

    # Addestra modelli
    vectorizer, dept_model, sent_model, metrics = train_models(df)

    # Avvia dashboard Tkinter (nessun token richiesto)
    app = ReviewDashboard(vectorizer, dept_model, sent_model, metrics)
    app.mainloop()


if __name__ == "__main__":
    main()
