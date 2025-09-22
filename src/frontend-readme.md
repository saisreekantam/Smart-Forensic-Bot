Key Files & Their Purpose
Root Files
.gitignore: Specifies files/folders to ignore in git.
components.json: Configuration for component imports (if using a design system).
eslint.config.js: ESLint configuration for code linting.
index.html: Main HTML template for the Vite app.
package.json: Project dependencies and scripts.
postcss.config.js: PostCSS configuration for CSS processing.
tailwind.config.ts: Tailwind CSS configuration.
tsconfig.json*: TypeScript configuration files.
vite.config.ts: Vite build tool configuration.
public/
favicon.ico: App icon.
placeholder.svg: Placeholder image.
robots.txt: Search engine crawling rules.

src/
Entry & Global Styles
App.tsx: Main React component, sets up routing/layout.
main.tsx: React entry point, renders <App />.
App.css, index.css: Global styles.
assets/
forensic-hero.jpg: Hero image for dashboard.
investigation-icon.png: Icon for investigation features.
components/
dashboard/
QueryInterface.tsx: Natural language query input for forensic searches.
RecentActivity.tsx: Displays recent forensic activities/events.
StatsCard.tsx: Card component for dashboard statistics.
layout/
AppSidebar.tsx: Sidebar navigation for the app.
Header.tsx: Top header bar.


Reusable UI primitives (buttons, cards, inputs, dialogs, etc.), mostly built with Tailwind CSS.
Examples:

button.tsx: Button component.
card.tsx: Card layout.
input.tsx: Input field.
badge.tsx: Status badge.
toast.tsx, toaster.tsx, use-toast.tsx: Toast notification system.
scroll-area.tsx: Scrollable area for chat/messages.
Many more: accordion, alert, avatar, calendar, dialog, dropdown, etc.
hooks/
use-mobile.tsx: Custom hook for mobile detection.
use-toast.ts: Custom hook for toast notifications.
lib/
utils.tsx: Utility functions used across the app.

pages/
Chatbot.tsx: Forensic AI chatbot interface. Allows users to interact with the forensic assistant, send queries, and receive responses.
Dashboard.tsx: Main dashboard page. Shows stats, upload UFDR files, run queries, view system status, and recent activity.
Index.tsx: Landing page or main entry point.
Login.tsx: User authentication/login page.
NotFound.tsx: 404 error page for unknown routes.
How It Works
Dashboard:

Upload UFDR files to backend for analysis.
View case statistics, evidence processed, queries executed, and reports generated.
Run natural language queries via the Query Interface.
See system status and recent activity.
Chatbot:

Chat with the forensic AI assistant.
Ask investigation questions, get instant responses.
Handles backend errors gracefully and shows connection status.
UI Components:

Built with Tailwind CSS for rapid styling.
Reusable primitives for consistent design.
Routing:

Each page (Dashboard, Chatbot, etc.) is a React component.
Navigation handled via sidebar/header.

For Collaborators
Add new features in the src/pages/ directory as new React components.
Use UI primitives from src/components/ui/ for consistent design.
Global styles are in App.css and index.css.
Backend endpoints are currently hardcoded to localhost (change as needed).
Assets (images/icons) go in src/assets/.
Utilities/hooks should be placed in src/lib/ and src/hooks/.