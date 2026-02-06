"""Flask app for user authentication (login, logout, register)."""
from flask import jsonify
from recommender import load_and_build_artifacts, recommend
# imports
from flask import Flask, request, redirect, render_template, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import bcrypt
import re
from config import Config


"""
Notes:
- Authentication flow based on a GeeksForGeeks tutorial.
- Password hashing uses bcrypt - From google
"""

app = Flask(__name__)
app.config.from_object(Config)  
mysql = MySQL(app)
ARTIFACTS = None

@app.route("/")
@app.route("/login", methods=["GET", "POST"])
def login():
    """Render login form and authenticate users on POST."""
    msg = ""
    #  check if login was submitted
    if request.method == "POST" and "email" in request.form and "password" in request.form:
        # normalize and extract
        email = request.form["email"].strip().lower()
        password = request.form["password"].encode("utf-8")
    # query db for user
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT id, email, password_hash FROM users WHERE email = %s",
        (email,))
        account = cursor.fetchone()
        cursor.close()
    # verify PW using bcrpyt
        if account and bcrypt.checkpw(password, account["password_hash"].encode("utf-8")):
            session["loggedin"] = True
            session["id"] = account["id"]
            session["email"] = account["email"]
            return render_template("index.html", msg="Logged in successfully!")
        else:
            msg = "Incorrect email/password!"

    return render_template("login.html", msg=msg)
# logout route
@app.route("/logout")
def logout():
    """Clear session and redirect to login."""
    session.clear()
    return redirect(url_for("login"))
# register route
@app.route("/register", methods=["GET", "POST"])
def register():
    """Render registration form and create a new user on POST."""
    msg = ""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password_raw = request.form.get("password", "")
        # validation
        if not email or not password_raw:
            return render_template("register.html", msg="Please fill out the form!")

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return render_template("register.html", msg="Invalid email address!")

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        existing = cursor.fetchone()

        if existing:
            cursor.close()
            return render_template("register.html", msg="Account already exists!")
        password_hash = bcrypt.hashpw(password_raw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        
        try:
            cursor.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s)",
            (email, password_hash),)
            mysql.connection.commit()
        except Exception as e:
            mysql.connection.rollback()
            cursor.close()
            return render_template("register.html", msg=f"Registration failed: {e}")
        finally:
            cursor.close()

        msg = "You have successfully registered! Please log in."
        return redirect(url_for("login"))

    return render_template("register.html", msg=msg)


@app.route("/preferences")
def preferences():
    """Render music preferences page for logged-in users."""
    if not session.get("loggedin"):
        return redirect(url_for("login"))
    return render_template("preferences.html")


@app.route("/recommendations", methods=["POST"])
def recommendations():
    """Return music recommendations for the provided preference payload."""
    if not session.get("loggedin"):
        return jsonify({"error": "Authentication required"}), 401

    data = request.get_json(silent=True) or {}
    prefs = data.get("preferences") or {}
    genres = data.get("genres") or None
    tempo_bpm = data.get("tempo_bpm")
    k = 1

    try:
        global ARTIFACTS
        if ARTIFACTS is None:
            ARTIFACTS = load_and_build_artifacts("music_dataset.csv")
        recs = recommend(
            artifacts=ARTIFACTS,
            preferences=prefs,
            genres=genres,
            tempo_bpm=tempo_bpm,
            k=1,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    minimal = [
        {
            "artist": row.get("artists"),
            "title": row.get("track_name"),
        }
        for _, row in recs.iterrows()
    ]

    if not minimal:
        return jsonify({"error": "No recommendations found"}), 404

    return jsonify(minimal[0])
    # return jsonify(recs.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
