"""flask for user authentication (login, logout, register)."""

from flask import Flask, request, redirect, render_template, url_for, session, jsonify
from flask_mysqldb import MySQL
import MySQLdb.cursors
import bcrypt
import re
from config import Config
from recommender import load_artifacts, recommend

app = Flask(__name__)
app.config.from_object(Config)
mysql = MySQL(app)

ARTIFACTS = None

# login route
@app.route("/")
@app.route("/login", methods=["GET", "POST"])
def login():
    """Render login form and authenticate users on POST."""
    msg = ""
    if request.method == "POST" and "email" in request.form and "password" in request.form:
        email = request.form["email"].strip().lower()
        password = request.form["password"].encode("utf-8")

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            "SELECT id, email, password_hash FROM users WHERE email = %s",
            (email,),
        )
        account = cursor.fetchone()
        cursor.close()

        if account and bcrypt.checkpw(password, account["password_hash"].encode("utf-8")):
            session["loggedin"] = True
            session["id"] = account["id"]
            session["email"] = account["email"]
            return render_template("index.html", msg="Logged in successfully!")
        else:
            msg = "Incorrect email/password!"

    return render_template("login.html", msg=msg)

# logout
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
        # verification/auth
        if not email or not password_raw:
            return render_template("register.html", msg="Please fill out the form!")

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return render_template("register.html", msg="Invalid email address!")

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # get id
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        existing = cursor.fetchone()

        if existing:
            cursor.close()
            return render_template("register.html", msg="Account already exists!")

        password_hash = bcrypt.hashpw(
            password_raw.encode("utf-8"),
            bcrypt.gensalt(),
        ).decode("utf-8")

        try:
            # insert user into mysql
            cursor.execute(
                "INSERT INTO users (email, password_hash) VALUES (%s, %s)",
                (email, password_hash),
            )
            mysql.connection.commit()
        except Exception as e:
            mysql.connection.rollback()
            cursor.close()
            return render_template("register.html", msg=f"Registration failed: {e}")
        finally:
            cursor.close()

        return redirect(url_for("login"))

    return render_template("register.html", msg=msg)

# preferences route
@app.route("/preferences")
def preferences():
    if not session.get("loggedin"):
        return redirect(url_for("login"))
    return render_template("preferences.html")

# recommendations route
@app.route("/recommendations", methods=["POST"])
def recommendations():
    if not session.get("loggedin"):
        return jsonify({"error": "auth required"}), 401

    data = request.get_json(silent=True) or {}

    
    genres = data.get("genres") or None
    k = int(data.get("k", 1)) 

    try:
        global ARTIFACTS
        if ARTIFACTS is None:

            ARTIFACTS = load_artifacts("music_dataset.csv")

        # recommend
        recs = recommend(
            artifacts=ARTIFACTS,
            genres=genres,
            k=k,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if recs.empty:
        return jsonify({"error": "No recommendations found"}), 404

    #return one
    row0 = recs.iloc[0]
    return jsonify({
        "artist": row0.get("artists"),
        "title": row0.get("track_name"),
    })


if __name__ == "__main__":
    app.run(debug=True)