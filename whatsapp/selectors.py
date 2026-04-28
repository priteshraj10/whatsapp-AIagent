"""
whatsapp/selectors.py -- All WhatsApp Web CSS selectors in one place.

If WhatsApp updates its DOM, change selectors here -- nothing else needs to change.
"""

SEL_QR_CANVAS     = "canvas"
SEL_SEARCH_BOX    = '[data-testid="chat-list-search"]'
SEL_SEARCH_BOX2   = 'input[aria-label="Search or start a new chat"]'
SEL_COMPOSE       = '[data-testid="conversation-compose-box-input"]'
SEL_COMPOSE2      = 'div[contenteditable="true"][data-tab="10"]'
SEL_MSG_CONTAINER = '[data-testid="msg-container"]'
SEL_MSG_PANEL     = '[data-testid="conversation-panel-messages"]'
SEL_MSG_PANEL_ALT = "#main .copyable-area"
SEL_HEADER        = 'header [data-testid="conversation-header"]'
SEL_FIRST_RESULT  = '[data-testid="cell-frame-container"], [role="listitem"]'

# Message Interaction Selectors
SEL_MSG_ROW       = '[data-testid="msg-container"]'
SEL_MSG_DROPDOWN  = '[data-testid="down-context"]'
SEL_REACT_MENU    = '[data-testid="reaction-menu"]'
SEL_DELETE_OPTION = 'li:has-text("Delete")'
SEL_DELETE_FOR_EVERYONE = 'button:has-text("Delete for everyone")'
SEL_OK_BUTTON = 'button:has-text("OK")'
