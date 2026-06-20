import streamlit as st
import json
from datetime import datetime
from pathlib import Path

SUPPORT_FILE = "support_tickets.json"

# ---------- Helpers ----------
def load_tickets():
    if not Path(SUPPORT_FILE).exists():
        return []
    try:
        with open(SUPPORT_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_tickets(tickets):
    with open(SUPPORT_FILE, "w") as f:
        json.dump(tickets, f, indent=2)

def next_ticket_id(tickets):
    return max([t["id"] for t in tickets], default=0) + 1

# ---------- Submit ----------
def submit_ticket(username, subject, description):
    tickets = load_tickets()
    ticket = {
        "id": next_ticket_id(tickets),
        "user": username,
        "subject": subject,
        "description": description,
        "status": "Open",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    tickets.append(ticket)
    save_tickets(tickets)

# ---------- UI ----------
def user_support_page():
    st.header("User Support")

    username = st.session_state.get("username")
    if not username:
        st.warning("Please login to use support.")
        return

    tab1, tab2 = st.tabs(["Submit Ticket", "My Tickets"])

    # Submit Ticket
    with tab1:
        with st.form("support_form"):
            subject = st.text_input("Subject")
            description = st.text_area("Description", height=150)
            submitted = st.form_submit_button("Submit")

            if submitted:
                if not subject or not description:
                    st.error("All fields are required.")
                else:
                    submit_ticket(username, subject, description)
                    st.success("Ticket submitted.")

    # View Tickets
    with tab2:
        tickets = [t for t in load_tickets() if t["user"] == username]

        if not tickets:
            st.info("No tickets submitted.")
            return

        for t in tickets:
            with st.expander(f"#{t['id']} | {t['subject']} | {t['status']}"):
                st.write(t["description"])
                st.caption(f"Created: {t['created']}")
