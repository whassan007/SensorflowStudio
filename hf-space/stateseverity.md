{
  "skill_name": "generate_statewide_ssam_gis",
  "version": "1.0.0",
  "description": "Transforms a static SSAM Safety data table into a high-performance, statewide GIS React dashboard.",
  "orchestration": {
    "framework": "antigravity",
    "execution_mode": "mcp-agentic",
    "target_model_temperature": 0.2
  },
  "dependencies": {
    "required_packages": [
      "@deck.gl/core",
      "@deck.gl/react",
      "@deck.gl/layers",
      "@deck.gl/geo-layers",
      "react-map-gl",
      "lucide-react"
    ]
  },
  "system_context": "You are a senior UI/UX architect specializing in autonomous vehicle safety systems, geospatial data visualization, and React. Your objective is to rewrite a static SSAM Safety component into a dark-mode geospatial dashboard capable of handling statewide California DMV disengagement and collision data.",
  "architectural_rules": [
    "1. Layout structure: Top Filter Bar -> Main Map Canvas (Center) -> Sliding Metadata Inspector (Right) -> Paginated Data Grid (Bottom).",
    "2. Use Deck.gl's H3HexagonLayer for high-level state clustering, and PathLayer/GeoJsonLayer for street-level polylines.",
    "3. Keep the styling aligned with Sensorflow Studio's dark mode aesthetic.",
    "4. Ensure the Severity Annotator is refactored into a collapsible side-drawer that triggers `onSelect` of a map layer."
  ],
  "expected_output": {
    "format": "tsx",
    "structure": "A complete, modular React functional component with mocked state hooks for map view state, search queries, and selected segment data."
  }
}