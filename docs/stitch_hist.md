# Stitch MCP Prompt for History UI

Copy and paste the following prompt into the `mcp_StitchMCP_generate_screen_from_text` tool, or copy it directly to your UI generator:

```text
Design a History Page for a Learning & AI Recommendation platform. The UI needs to display a vertical timeline of AI explanation events and drift tracking. 

Requirements:
1. **Layout Context**: The page should fit within a standard dashboard layout (assume a sidebar is on the left, so this fills the right main content area). Keep the design sleek, using a modern Light or Dark theme with glassmorphism or soft drop shadows.
2. **Timeline View**: The main content should be a vertical chronological timeline. Each node in the timeline represents a "Snapshot" from our `consistency_store` backend.
3. **Data Representation (Cards per Node)**:
   - **Timestamp / Date**: Clearly label when the AI made the recommendation.
   - **Feature Importance (SHAP)**: Some nodes should show a miniature bar chart or badge group representing "Top Features" for that snapshot (e.g., Score: 85%, Study Time: 120min).
   - **Drift Alerts**: If there is a shift in the model's behavior, display a distinct warning or badge indicating "Feature Drift Detected" with a highlighting color (e.g., amber or red).
4. **Role Toggle/State**: Include a sleek sticky header that says "Explanation History". Below it, show a dropdown or search bar (intended for the Instructor role to select a specific student ID), which is disabled or hidden for pure Student roles.
5. **Aesthetics**: Use modern styling (Tailwind-like utility approach). Apply polished typography (Inter/Roboto), subtle gradients on key elements, and hover micro-interactions on the timeline cards.
```
