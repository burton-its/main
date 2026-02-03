# imports
from flask import Flask, request, redirect, render_template, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import bcrypt
import re
from config import Config


"""
template from geeksforgeeks
bcrypt used with help from ChatGPT
"""

app = Flask(__name__)
app.config.from_object(Config)  
mysql = MySQL(app)

@app.route("/")
@app.route("/login", methods=["GET", "POST"])
def login():
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
    session.clear()
    return redirect(url_for("login"))
# register route
@app.route("/register", methods=["GET", "POST"])
def register():
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

if __name__ == "__main__":
    app.run(debug=True)