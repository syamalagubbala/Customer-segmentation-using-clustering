"""
Customer Segmentation using Clustering (K-Means)
Flask backend + HTML interface.

Default logins (change them after first run!):
    Admin login : admin / admin123
    User login  : user / user123   or   analyst / analyst123
Forgot password uses an OTP sent by email (SMTP) or SMS (Twilio) - see README.
"""
import base64
import json
import os
import re
import secrets
import smtplib
import sqlite3
import urllib.parse
import urllib.request
from email.message import EmailMessage
from datetime import datetime, timedelta
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import (Flask, flash, redirect, render_template, request,
                   send_file, session, url_for)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_env_file():
    """Tiny .env loader (no extra package needed)."""
    path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_env_file()
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "users.db")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.joblib")
META_PATH = os.path.join(DATA_DIR, "meta.json")
SAMPLE_PATH = os.path.join(BASE_DIR, "sample_data", "customers_sample.csv")
CLUSTER_COL = "Cluster"

os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key-in-production")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB uploads


# --------------------------------------------------------------------------
# Database / authentication
# --------------------------------------------------------------------------
OTP_VALID_MINUTES = 10
OTP_MAX_ATTEMPTS = 5
OTP_RESEND_SECONDS = 30
DEFAULT_COUNTRY_CODE = os.environ.get("DEFAULT_COUNTRY_CODE", "+91")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_mobile(text):
    """Return mobile in +<countrycode><number> format, or None if invalid."""
    raw = re.sub(r"[^\d+]", "", text or "")
    if not raw:
        return None
    if raw.startswith("+"):
        num = raw
    elif len(raw) == 10:
        num = DEFAULT_COUNTRY_CODE + raw
    else:
        num = "+" + raw.lstrip("0")
    return num if re.fullmatch(r"\+\d{10,15}", num) else None


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   username TEXT UNIQUE NOT NULL,
                   password_hash TEXT NOT NULL,
                   role TEXT NOT NULL DEFAULT 'user',
                   email TEXT,
                   mobile TEXT)"""
        )
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)")]
        for col in ("email", "mobile"):  # migrate older databases
            if col not in cols:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_mobile ON users(mobile)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS password_resets (
                   user_id INTEGER PRIMARY KEY,
                   otp_hash TEXT NOT NULL,
                   expires_at TEXT NOT NULL,
                   created_at TEXT NOT NULL,
                   attempts INTEGER NOT NULL DEFAULT 0)"""
        )

        # username, password, role, env prefix, placeholder email, placeholder mobile
        defaults = [
            ("admin", "admin123", "admin", "ADMIN", "admin@example.com", "+910000000001"),
            ("user", "user123", "user", "USER", "user@example.com", "+910000000002"),
            ("analyst", "analyst123", "user", "ANALYST", "analyst@example.com", "+910000000003"),
        ]
        for u, p, r, prefix, d_email, d_mobile in defaults:
            email = os.environ.get(f"{prefix}_EMAIL") or None
            mobile = normalize_mobile(os.environ.get(f"{prefix}_MOBILE", "")) or None
            row = conn.execute("SELECT id, email, mobile FROM users WHERE username=?", (u,)).fetchone()
            if not row:
                conn.execute(
                    "INSERT INTO users (username, password_hash, role, email, mobile) VALUES (?,?,?,?,?)",
                    (u, generate_password_hash(p), r, email or d_email, mobile or d_mobile),
                )
            else:
                new_email = email or row["email"] or d_email
                new_mobile = mobile or row["mobile"] or d_mobile
                conn.execute("UPDATE users SET email=?, mobile=? WHERE id=?", (new_email, new_mobile, row["id"]))


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def handle_login(expected_role):
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with db() as conn:
            row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row and check_password_hash(row["password_hash"], password):
            if row["role"] != expected_role:
                other = "Admin" if row["role"] == "admin" else "User"
                flash(f"This account is not a {expected_role} account. Please use the {other} Login.", "danger")
            else:
                session["user"] = row["username"]
                session["role"] = row["role"]
                return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "danger")
    return render_template("login.html", role=expected_role)


@app.route("/login", methods=["GET", "POST"])
def login():
    return handle_login("user")


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    return handle_login("admin")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        mobile = normalize_mobile(request.form.get("mobile", ""))
        password = request.form.get("password", "")
        if len(username) < 3 or len(password) < 6:
            flash("Username needs 3+ characters and password 6+ characters.", "danger")
        elif not EMAIL_RE.match(email):
            flash("Please enter a valid email address.", "danger")
        elif not mobile:
            flash("Please enter a valid mobile number (10 digits, or with country code like +919876543210).", "danger")
        else:
            try:
                with db() as conn:
                    conn.execute(
                        "INSERT INTO users (username, password_hash, role, email, mobile) VALUES (?,?, 'user', ?, ?)",
                        (username, generate_password_hash(password), email, mobile),
                    )
                flash("Account created. You can log in now.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That username, email or mobile number is already registered.", "danger")
    return render_template("register.html")


# ---------------- OTP delivery (email / SMS) ----------------
def email_configured():
    return all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS"))


def sms_configured():
    return all(os.environ.get(k) for k in ("TWILIO_SID", "TWILIO_TOKEN", "TWILIO_FROM"))


def send_email(to, subject, body):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("SMTP_FROM") or os.environ["SMTP_USER"]
    msg["To"] = to
    msg.set_content(body)
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=20) as server:
            server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls()
            server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
            server.send_message(msg)


def send_sms(to, body):
    sid, token = os.environ["TWILIO_SID"], os.environ["TWILIO_TOKEN"]
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    data = urllib.parse.urlencode({"To": to, "From": os.environ["TWILIO_FROM"], "Body": body}).encode()
    req = urllib.request.Request(url, data=data)
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{sid}:{token}".encode()).decode())
    urllib.request.urlopen(req, timeout=20).read()


def deliver_otp(user, method, otp):
    text = (f"Your Customer Segmentation password reset OTP is {otp}. "
            f"It is valid for {OTP_VALID_MINUTES} minutes. Do not share it with anyone.")
    if method == "email":
        if email_configured():
            send_email(user["email"], "Your password reset OTP", text)
            return
        target = user["email"]
    else:
        if sms_configured():
            send_sms(user["mobile"], text)
            return
        target = user["mobile"]
    # Development fallback: SMTP / Twilio not configured -> print in the terminal
    print(f"\n[DEV MODE] OTP for '{user['username']}' ({method}: {target}) = {otp}\n", flush=True)


def find_user(identifier, role):
    ident = identifier.strip()
    mobile = normalize_mobile(ident) if "@" not in ident else None
    with db() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE role=? AND (lower(username)=lower(?) OR lower(email)=lower(?) OR mobile=?)",
            (role, ident, ident, mobile or "__none__"),
        ).fetchone()


def issue_otp(identifier, role, method):
    """Returns 'sent', 'none' (no matching account), 'cooldown' or 'error'."""
    user = find_user(identifier, role)
    if not user or not user["email" if method == "email" else "mobile"]:
        return "none"
    now = datetime.now()
    with db() as conn:
        old = conn.execute("SELECT created_at FROM password_resets WHERE user_id=?", (user["id"],)).fetchone()
        if old and (now - datetime.fromisoformat(old["created_at"])).total_seconds() < OTP_RESEND_SECONDS:
            return "cooldown"
        otp = f"{secrets.randbelow(1000000):06d}"
        conn.execute(
            "INSERT OR REPLACE INTO password_resets (user_id, otp_hash, expires_at, created_at, attempts) "
            "VALUES (?,?,?,?,0)",
            (user["id"], generate_password_hash(otp),
             (now + timedelta(minutes=OTP_VALID_MINUTES)).isoformat(), now.isoformat()),
        )
    try:
        deliver_otp(user, method, otp)
    except Exception as e:  # noqa: BLE001
        print(f"[OTP ERROR] {e}", flush=True)
        with db() as conn:
            conn.execute("DELETE FROM password_resets WHERE user_id=?", (user["id"],))
        return "error"
    return "sent"


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    role = request.values.get("role", "user")
    if role not in ("user", "admin"):
        role = "user"
    login_endpoint = "admin_login" if role == "admin" else "login"
    ctx = {"role": role, "login_endpoint": login_endpoint, "step": 1, "method": "email",
           "dev_mode": False, "valid_minutes": OTP_VALID_MINUTES}

    if request.method == "GET":
        for k in ("reset_ident", "reset_method", "reset_role"):
            session.pop(k, None)
        return render_template("forgot.html", **ctx)

    action = request.form.get("action")
    if action in ("send", "resend"):
        if action == "resend":
            ident, method = session.get("reset_ident", ""), session.get("reset_method", "email")
        else:
            ident = request.form.get("identifier", "").strip()
            method = request.form.get("method", "email")
        if not ident or method not in ("email", "sms"):
            flash("Please enter your username, email or mobile number.", "danger")
            return render_template("forgot.html", **ctx)
        result = issue_otp(ident, role, method)
        if result == "error":
            flash("Could not send the OTP. Please check the email/SMS settings and try again.", "danger")
            return render_template("forgot.html", **ctx)
        if result == "cooldown":
            flash(f"Please wait {OTP_RESEND_SECONDS} seconds before requesting another OTP.", "warning")
        else:
            flash("If an account matches, an OTP has been sent to its registered "
                  + ("email address" if method == "email" else "mobile number") + ".", "info")
        session.update(reset_ident=ident, reset_method=method, reset_role=role)
        ctx.update(step=2, method=method,
                   dev_mode=not (email_configured() if method == "email" else sms_configured()))
        return render_template("forgot.html", **ctx)

    if action == "reset":
        ident, method = session.get("reset_ident"), session.get("reset_method", "email")
        ctx.update(step=2, method=method,
                   dev_mode=not (email_configured() if method == "email" else sms_configured()))
        if not ident or session.get("reset_role") != role:
            flash("Session expired. Please start again.", "warning")
            ctx.update(step=1)
            return render_template("forgot.html", **ctx)
        otp = request.form.get("otp", "").strip()
        new_pw = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")
        if len(new_pw) < 6:
            flash("New password must be at least 6 characters.", "danger")
            return render_template("forgot.html", **ctx)
        if new_pw != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("forgot.html", **ctx)

        user = find_user(ident, role)
        ok = False
        if user:
            with db() as conn:
                rec = conn.execute("SELECT * FROM password_resets WHERE user_id=?", (user["id"],)).fetchone()
                if rec and datetime.fromisoformat(rec["expires_at"]) > datetime.now() \
                        and rec["attempts"] < OTP_MAX_ATTEMPTS:
                    if check_password_hash(rec["otp_hash"], otp):
                        ok = True
                        conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                                     (generate_password_hash(new_pw), user["id"]))
                        conn.execute("DELETE FROM password_resets WHERE user_id=?", (user["id"],))
                    else:
                        conn.execute("UPDATE password_resets SET attempts = attempts + 1 WHERE user_id=?",
                                     (user["id"],))
        if ok:
            for k in ("reset_ident", "reset_method", "reset_role"):
                session.pop(k, None)
            flash("Password reset successful. Please log in with your new password.", "success")
            return redirect(url_for(login_endpoint))
        flash("Invalid or expired OTP. You can request a new one.", "danger")
        return render_template("forgot.html", **ctx)

    return render_template("forgot.html", **ctx)


@app.route("/logout")
def logout():
    was_admin = session.get("role") == "admin"
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("admin_login" if was_admin else "login"))


# --------------------------------------------------------------------------
# Data / model helpers
# --------------------------------------------------------------------------
def read_csv_flexible(file_or_path):
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            return pd.read_csv(file_or_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode CSV file.")


def load_df():
    if not os.path.exists(DATASET_PATH):
        return None
    return pd.read_csv(DATASET_PATH)


def save_df(df):
    df.to_csv(DATASET_PATH, index=False)


def load_meta():
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH) as f:
        return json.load(f)


def save_meta(meta):
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def numeric_columns(df):
    cols = []
    for c in df.columns:
        if c == CLUSTER_COL:
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().mean() >= 0.9 and converted.nunique() > 1:
            cols.append(c)
    return cols


def segment_names(df, features):
    """Create human-readable names by comparing each cluster mean to overall mean."""
    overall_mean = df[features].mean()
    overall_std = df[features].std().replace(0, 1).fillna(1)
    names = {}
    for c, g in df.groupby(CLUSTER_COL):
        z = (g[features].mean() - overall_mean) / overall_std
        parts = []
        for f in z.abs().sort_values(ascending=False).index:
            if z[f] >= 0.5:
                parts.append(f"High {f}")
            elif z[f] <= -0.5:
                parts.append(f"Low {f}")
        label = ", ".join(parts[:3]) if parts else "Average customers"
        names[str(int(c))] = f"Segment {int(c) + 1}: {label}"
    return names


def train_model(df, features, k, auto_k, id_col):
    X = df[features].apply(pd.to_numeric, errors="coerce")
    medians = X.median()
    X = X.fillna(medians)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    max_k = min(10, len(df) - 1)
    scores = []
    for kk in range(2, max_k + 1):
        km = KMeans(n_clusters=kk, n_init=10, random_state=42).fit(Xs)
        scores.append({
            "k": kk,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(Xs, km.labels_)),
        })
    if auto_k:
        k = max(scores, key=lambda s: s["silhouette"])["k"]
    k = max(2, min(int(k), max_k))

    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    df = df.copy()
    df[CLUSTER_COL] = km.labels_
    df[features] = X  # store imputed numeric values
    names = segment_names(df, features)

    joblib.dump({"scaler": scaler, "kmeans": km, "features": features, "medians": medians.to_dict()}, MODEL_PATH)
    save_df(df)
    meta = {
        "features": features,
        "k": k,
        "auto_k": bool(auto_k),
        "id_col": id_col,
        "scores": scores,
        "segment_names": names,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset_name": load_meta().get("dataset_name", "dataset.csv"),
    }
    save_meta(meta)
    return meta


def predict_rows(values_df):
    """values_df: DataFrame with the model's feature columns -> cluster labels, distances."""
    bundle = joblib.load(MODEL_PATH)
    X = values_df[bundle["features"]].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(pd.Series(bundle["medians"]))
    Xs = bundle["scaler"].transform(X)
    km = bundle["kmeans"]
    labels = km.predict(Xs)
    dists = km.transform(Xs)
    return labels, dists


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
@app.context_processor
def inject_globals():
    return {"now_year": datetime.now().year}


@app.route("/")
@login_required
def dashboard():
    df = load_df()
    meta = load_meta()
    stats = None
    if df is not None and CLUSTER_COL in df.columns and meta:
        feats = meta["features"]
        Xs = joblib.load(MODEL_PATH)["scaler"].transform(df[feats])
        sil = silhouette_score(Xs, df[CLUSTER_COL]) if df[CLUSTER_COL].nunique() > 1 else 0
        sizes = df[CLUSTER_COL].value_counts().sort_index()
        stats = {
            "total": len(df),
            "k": meta["k"],
            "features": len(feats),
            "silhouette": round(float(sil), 3),
            "segments": [
                {"name": meta["segment_names"][str(int(c))], "count": int(n),
                 "pct": round(100 * n / len(df), 1)}
                for c, n in sizes.items()
            ],
        }
    return render_template("dashboard.html", df_loaded=df is not None, meta=meta, stats=stats)


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        try:
            if request.form.get("use_sample"):
                df = read_csv_flexible(SAMPLE_PATH)
                name = "customers_sample.csv"
            else:
                f = request.files.get("file")
                if not f or not f.filename:
                    flash("Please choose a CSV file.", "danger")
                    return redirect(url_for("upload"))
                if not f.filename.lower().endswith(".csv"):
                    flash("Only .csv files are supported.", "danger")
                    return redirect(url_for("upload"))
                df = read_csv_flexible(f)
                name = f.filename
        except Exception as e:  # noqa: BLE001
            flash(f"Could not read the file: {e}", "danger")
            return redirect(url_for("upload"))

        df = df.drop(columns=[CLUSTER_COL], errors="ignore").dropna(how="all")
        df.columns = [str(c).strip() for c in df.columns]
        if len(df) < 5:
            flash("The dataset needs at least 5 rows.", "danger")
            return redirect(url_for("upload"))
        if len(numeric_columns(df)) < 2:
            flash("The dataset needs at least 2 numeric columns for clustering.", "danger")
            return redirect(url_for("upload"))

        save_df(df)
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        save_meta({"dataset_name": name})
        flash(f"Dataset '{name}' uploaded ({len(df)} rows). Now choose features and run clustering.", "success")
        return redirect(url_for("configure"))
    return render_template("upload.html")


@app.route("/configure", methods=["GET", "POST"])
@login_required
def configure():
    df = load_df()
    if df is None:
        flash("Upload a dataset first.", "warning")
        return redirect(url_for("upload"))
    num_cols = numeric_columns(df)
    meta = load_meta()

    if request.method == "POST":
        features = request.form.getlist("features")
        id_col = request.form.get("id_col") or ""
        auto_k = request.form.get("k_mode") == "auto"
        try:
            k = int(request.form.get("k", 4))
        except ValueError:
            k = 4
        if len(features) < 2:
            flash("Select at least 2 features.", "danger")
            return redirect(url_for("configure"))
        train_model(df, features, k, auto_k, id_col)
        flash("Clustering complete!", "success")
        return redirect(url_for("analytics"))

    preview = df.head(10).to_html(classes="table", index=False, border=0)
    return render_template(
        "configure.html", preview=preview, columns=list(df.columns), num_cols=num_cols,
        rows=len(df), meta=meta,
        # guess an ID column: name contains "id"
        guess_id=next((c for c in df.columns if "id" in c.lower()), ""),
        selected=meta.get("features", [c for c in num_cols if "id" not in c.lower()][:3]),
    )


@app.route("/customers")
@login_required
def customers():
    df = load_df()
    meta = load_meta()
    if df is None or CLUSTER_COL not in df.columns:
        flash("Run clustering first.", "warning")
        return redirect(url_for("configure" if df is not None else "upload"))

    q = request.args.get("q", "").strip()
    cl = request.args.get("cluster", "")
    page = max(1, request.args.get("page", 1, type=int))
    per_page = 20

    view = df.copy()
    view.insert(0, "_rid", range(len(view)))
    if cl != "":
        view = view[view[CLUSTER_COL] == int(cl)]
    if q:
        mask = view.astype(str).apply(lambda col: col.str.contains(q, case=False, na=False)).any(axis=1)
        view = view[mask]

    total = len(view)
    pages = max(1, int(np.ceil(total / per_page)))
    page = min(page, pages)
    chunk = view.iloc[(page - 1) * per_page: page * per_page]
    names = meta["segment_names"]
    rows = []
    for _, r in chunk.iterrows():
        rows.append({
            "rid": int(r["_rid"]),
            "cells": [("" if pd.isna(r[c]) else r[c]) for c in df.columns if c != CLUSTER_COL],
            "cluster": int(r[CLUSTER_COL]),
            "segment": names[str(int(r[CLUSTER_COL]))],
        })
    return render_template(
        "customers.html", columns=[c for c in df.columns if c != CLUSTER_COL], rows=rows,
        total=total, page=page, pages=pages, q=q, cl=cl, names=names,
    )


@app.route("/customers/add", methods=["GET", "POST"])
@login_required
def add_customer():
    df = load_df()
    meta = load_meta()
    if df is None or not meta.get("features"):
        flash("Run clustering first.", "warning")
        return redirect(url_for("upload"))
    features = meta["features"]
    id_col = meta.get("id_col", "")
    cols = [c for c in df.columns if c != CLUSTER_COL]

    if request.method == "POST":
        new = {}
        for c in cols:
            val = request.form.get(f"col_{c}", "").strip()
            new[c] = val
        try:
            for f in features:
                new[f] = float(new[f])
        except ValueError:
            flash("All feature fields must be valid numbers.", "danger")
            return redirect(url_for("add_customer"))
        for c in cols:  # convert other numeric columns when possible
            if c not in features and new[c] != "" and pd.api.types.is_numeric_dtype(df[c]):
                try:
                    new[c] = float(new[c])
                except ValueError:
                    pass
            elif new[c] == "":
                new[c] = np.nan
        label, _ = predict_rows(pd.DataFrame([new]))
        new[CLUSTER_COL] = int(label[0])
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        save_df(df)
        flash(f"Customer added and assigned to {meta['segment_names'][str(int(label[0]))]}.", "success")
        return redirect(url_for("customers"))

    # build field definitions
    fields = []
    for c in cols:
        f = {"name": c, "is_feature": c in features, "type": "text", "options": [], "value": ""}
        if c in features or pd.api.types.is_numeric_dtype(df[c]):
            f["type"] = "number"
        elif df[c].nunique() <= 10:
            f["type"] = "select"
            f["options"] = sorted(df[c].dropna().astype(str).unique())
        if c == id_col and pd.api.types.is_numeric_dtype(df[c]):
            f["value"] = int(df[c].max()) + 1
        fields.append(f)
    return render_template("add_customer.html", fields=fields)


@app.route("/customers/delete/<int:rid>", methods=["POST"])
@login_required
def delete_customer(rid):
    df = load_df()
    if df is not None and 0 <= rid < len(df):
        df = df.drop(index=rid).reset_index(drop=True)
        save_df(df)
        flash("Customer deleted.", "info")
    return redirect(request.referrer or url_for("customers"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    meta = load_meta()
    df = load_df()
    if not meta.get("features") or df is None or CLUSTER_COL not in df.columns:
        flash("Run clustering first.", "warning")
        return redirect(url_for("upload"))
    features = meta["features"]
    result = None
    values = {}
    if request.method == "POST":
        try:
            values = {f: float(request.form.get(f"f_{f}", "")) for f in features}
        except ValueError:
            flash("Please enter valid numbers in all fields.", "danger")
            return render_template("predict.html", features=features, result=None, values={})
        labels, dists = predict_rows(pd.DataFrame([values]))
        c = int(labels[0])
        d = dists[0]
        inv = 1 / (d + 1e-9)
        conf = inv / inv.sum()
        prof = df[df[CLUSTER_COL] == c][features].mean().round(2).to_dict()
        overall = df[features].mean().round(2).to_dict()
        result = {
            "cluster": c,
            "name": meta["segment_names"][str(c)],
            "confidence": round(float(conf[c]) * 100, 1),
            "profile": [{"feature": f, "you": values[f], "segment": prof[f], "overall": overall[f]} for f in features],
            "others": sorted(
                [{"name": meta["segment_names"][str(i)], "distance": round(float(x), 2)} for i, x in enumerate(d)],
                key=lambda r: r["distance"]),
        }
    return render_template("predict.html", features=features, result=result, values=values)


@app.route("/analytics")
@login_required
def analytics():
    df = load_df()
    meta = load_meta()
    if df is None or CLUSTER_COL not in df.columns or not meta.get("features"):
        flash("Run clustering first.", "warning")
        return redirect(url_for("configure" if df is not None else "upload"))

    feats = meta["features"]
    names = meta["segment_names"]
    bundle = joblib.load(MODEL_PATH)
    Xs = bundle["scaler"].transform(df[feats])
    labels = df[CLUSTER_COL].values
    clusters = sorted(df[CLUSTER_COL].unique())

    pca = PCA(n_components=2, random_state=42).fit(Xs)
    pts = pca.transform(Xs)
    cents = pca.transform(bundle["kmeans"].cluster_centers_)
    idx = np.random.RandomState(0).permutation(len(df))[:1500]
    scatter = {
        int(c): [{"x": round(float(pts[i, 0]), 3), "y": round(float(pts[i, 1]), 3)}
                 for i in idx if labels[i] == c]
        for c in clusters
    }
    overall_mean, overall_std = df[feats].mean(), df[feats].std().replace(0, 1).fillna(1)
    profile_means = df.groupby(CLUSTER_COL)[feats].mean()
    z = ((profile_means - overall_mean) / overall_std).round(3)
    sizes = df[CLUSTER_COL].value_counts().sort_index()

    table = []
    for c in clusters:
        table.append({
            "name": names[str(int(c))], "count": int(sizes[c]),
            "pct": round(100 * sizes[c] / len(df), 1),
            "means": [round(float(profile_means.loc[c, f]), 2) for f in feats],
        })

    # categorical breakdown (e.g. Gender) for low-cardinality text columns
    cat_cols = [c for c in df.columns if c not in feats and c != CLUSTER_COL
                and not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 8]
    cat_breakdown = {}
    for cc in cat_cols[:2]:
        ct = pd.crosstab(df[CLUSTER_COL], df[cc])
        cat_breakdown[cc] = {
            "labels": [f"Segment {int(c) + 1}" for c in ct.index],
            "datasets": [{"label": str(col), "data": [int(v) for v in ct[col]]} for col in ct.columns],
        }

    payload = {
        "labels": [f"Segment {int(c) + 1}" for c in clusters],
        "sizes": [int(sizes[c]) for c in clusters],
        "scatter": {str(k): v for k, v in scatter.items()},
        "centroids": [{"x": round(float(cents[i, 0]), 3), "y": round(float(cents[i, 1]), 3)} for i in range(len(cents))],
        "features": feats,
        "z": {str(int(c)): [float(v) for v in z.loc[c]] for c in clusters},
        "scores": meta["scores"],
        "cat": cat_breakdown,
        "k": meta["k"],
    }
    metrics = {
        "silhouette": round(float(silhouette_score(Xs, labels)), 3),
        "davies_bouldin": round(float(davies_bouldin_score(Xs, labels)), 3),
        "inertia": round(float(bundle["kmeans"].inertia_), 1),
        "variance_explained": round(float(pca.explained_variance_ratio_.sum()) * 100, 1),
    }
    biggest = max(table, key=lambda r: r["count"])
    smallest = min(table, key=lambda r: r["count"])
    insights = [
        f"Largest segment: {biggest['name']} with {biggest['count']} customers ({biggest['pct']}%).",
        f"Smallest segment: {smallest['name']} with {smallest['count']} customers ({smallest['pct']}%).",
        f"Silhouette score of {metrics['silhouette']} indicates "
        + ("strong" if metrics["silhouette"] > 0.5 else "reasonable" if metrics["silhouette"] > 0.25 else "weak")
        + " cluster separation.",
    ]
    return render_template("analytics.html", payload=payload, metrics=metrics, table=table,
                           features=feats, meta=meta, insights=insights, cat_cols=list(cat_breakdown))


@app.route("/download")
@login_required
def download():
    if not os.path.exists(DATASET_PATH):
        flash("No dataset available.", "warning")
        return redirect(url_for("dashboard"))
    df = load_df()
    meta = load_meta()
    if CLUSTER_COL in df.columns and meta.get("segment_names"):
        df["Segment"] = df[CLUSTER_COL].astype(str).map(meta["segment_names"])
    out = os.path.join(DATA_DIR, "export.csv")
    df.to_csv(out, index=False)
    return send_file(out, as_attachment=True, download_name="segmented_customers.csv")


@app.route("/reset", methods=["POST"])
@login_required
def reset():
    for p in (DATASET_PATH, MODEL_PATH, META_PATH):
        if os.path.exists(p):
            os.remove(p)
    flash("Dataset and model cleared.", "info")
    return redirect(url_for("upload"))


init_db()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
