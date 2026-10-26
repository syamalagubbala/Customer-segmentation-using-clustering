# Customer Segmentation using K-Means Clustering

Flask (Python) backend + HTML/CSS/JS interface.

## Features
- Separate Admin and User logins, register, forgot password via OTP (email / SMS) (hashed, SQLite)
- Upload any customer CSV (or use the bundled sample)
- Choose features, automatic or manual K (silhouette-based), run K-Means
- Customers table: search, filter by segment, delete, download CSV
- Add new customers - automatically assigned to a segment
- Predict a segment for any set of values (without saving)
- Analytics: segment sizes, PCA cluster map, segment profiles, elbow/silhouette, metrics, insights

## Run
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Login credentials
**Admin Login** (http://127.0.0.1:5000/admin/login)

| Username | Password |
|----------|----------|
| admin    | admin123 |

**User Login** (http://127.0.0.1:5000/login)

| Username | Password   |
|----------|------------|
| user     | user123    |
| analyst  | analyst123 |

New users can register from the User Login page (they enter email and mobile for OTP recovery).
## Forgot password (OTP)
Login page -> **Forgot password?** -> enter **username, email or mobile number** -> choose **Email** or **Mobile (SMS)** -> enter the 6-digit OTP + new password.
OTP is valid for 10 minutes, max 5 wrong attempts, 30 seconds between resends. Separate flow for Admin and User login.

### Make OTP really arrive (email / SMS)
1. Copy `.env.example` to `.env` and fill the values.
2. **Email:** any SMTP server. For Gmail enable 2-Step Verification and create an *App password*; put it in `SMTP_PASS`.
3. **SMS:** create a Twilio account and fill `TWILIO_SID`, `TWILIO_TOKEN`, `TWILIO_FROM`.
4. The built-in demo accounts have placeholder email/mobile. Put real ones in `.env` (`ADMIN_EMAIL`, `ADMIN_MOBILE`, `USER_EMAIL`, `USER_MOBILE`) to receive OTPs. New users enter their own email and mobile while registering.

**Without `.env` (development mode):** the app still works - the OTP is printed in the terminal where `python app.py` is running.

Set a real secret key before deploying: `set SECRET_KEY=your-secret` (Windows) / `export SECRET_KEY=...`.

## CSV format
Header row + at least 2 numeric columns, e.g.
`CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)`

## Structure
```
app.py              Flask app, ML logic, routes
templates/          HTML pages (Jinja2)
static/style.css    Styling
sample_data/        Sample CSV
data/               Generated at runtime (dataset, model, users DB)
```
Charts use Chart.js from a CDN, so the browser needs internet access for the analytics page.
