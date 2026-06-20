import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from app_config import SUPPORT_FILE

# ---------- Helpers ----------
def load_tickets():
    path = Path(SUPPORT_FILE)
    if not path.exists():
        return []
    try:
        tickets = json.loads(path.read_text(encoding="utf-8"))
        return tickets if isinstance(tickets, list) else []
    except (OSError, json.JSONDecodeError):
        return []

def save_tickets(tickets):
    path = Path(SUPPORT_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(tickets, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)

def next_ticket_id(tickets):
    valid_ids = [
        ticket.get("id", 0)
        for ticket in tickets
        if isinstance(ticket, dict) and isinstance(ticket.get("id", 0), int)
    ]
    return max(valid_ids, default=0) + 1

# ---------- Submit ----------
def submit_ticket(username, subject, description):
    tickets = load_tickets()
    ticket = {
        "id": next_ticket_id(tickets),
        "user": username.strip(),
        "subject": subject.strip(),
        "description": description.strip(),
        "status": "Open",
        "created": datetime.now(timezone.utc).isoformat(timespec="minutes"),
    }
    tickets.append(ticket)
    save_tickets(tickets)
    return ticket

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
                if not subject.strip() or not description.strip():
                    st.error("All fields are required.")
                else:
                    submit_ticket(username, subject, description)
                    st.success("Ticket submitted.")

    # View Tickets
    with tab2:
        tickets = [
            ticket
            for ticket in load_tickets()
            if isinstance(ticket, dict) and ticket.get("user") == username
        ]

        if not tickets:
            st.info("No tickets submitted.")
            return

        for ticket in tickets:
            label = (
                f"#{ticket.get('id', '?')} | "
                f"{ticket.get('subject', 'Untitled')} | "
                f"{ticket.get('status', 'Open')}"
            )
            with st.expander(label):
                st.write(ticket.get("description", ""))
                st.caption(f"Created: {ticket.get('created', 'Unknown')}")
