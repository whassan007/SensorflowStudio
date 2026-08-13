import { useMemo, useState } from 'react';
import DeckGL from '@deck.gl/react';
import { TileLayer } from '@deck.gl/geo-layers';
import { BitmapLayer, ScatterplotLayer } from '@deck.gl/layers';
import CircularProgress from '@mui/material/CircularProgress';
import { useFilters } from '../../context/FilterContext';
import { GeoJSONFeature, SEVERITY_COLORS, SEVERITY_RGB } from '../../types';

const INITIAL_VIEW_STATE = {
  longitude: -119.4,
  latitude: 36.6,
  zoom: 5.3,
  pitch: 0,
  bearing: 0,
};

interface HoverInfo {
  x: number;
  y: number;
  feature: GeoJSONFeature;
}

interface PickInfo {
  x: number;
  y: number;
  object?: GeoJSONFeature;
}

export default function MapCanvas() {
  const { data, loading, selected, setSelected } = useFilters();
  const [hover, setHover] = useState<HoverInfo | null>(null);

  const features = data?.geojson.features ?? [];

  const layers = useMemo(() => {
    const basemap = new TileLayer({
      id: 'carto-dark-basemap',
      data: 'https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
      minZoom: 0,
      maxZoom: 19,
      tileSize: 256,
      renderSubLayers: (props: {
        tile: { bbox: { west: number; south: number; east: number; north: number } };
        data: unknown;
      }) => {
        const { west, south, east, north } = props.tile.bbox;
        return new BitmapLayer(props, {
          data: undefined,
          image: props.data,
          bounds: [west, south, east, north],
        });
      },
    });

    const points = new ScatterplotLayer({
      id: 'ssam-conflicts',
      data: features,
      pickable: true,
      stroked: true,
      radiusUnits: 'pixels',
      lineWidthUnits: 'pixels',
      getPosition: (f: GeoJSONFeature) => f.geometry.coordinates,
      getRadius: (f: GeoJSONFeature) => 4 + f.properties.severity_index * 10,
      getFillColor: (f: GeoJSONFeature) => {
        const [r, g, b] = SEVERITY_RGB[f.properties.severity_label] ?? [128, 128, 128];
        return [r, g, b, 200];
      },
      getLineColor: (f: GeoJSONFeature) =>
        selected && f.properties.id === selected.id ? [255, 255, 255, 255] : [0, 0, 0, 120],
      getLineWidth: (f: GeoJSONFeature) => (selected && f.properties.id === selected.id ? 3 : 1),
      onHover: (info: PickInfo) =>
        setHover(info.object ? { x: info.x, y: info.y, feature: info.object } : null),
      onClick: (info: PickInfo) => {
        const feature = info.object;
        if (!feature) return;
        const [lng, lat] = feature.geometry.coordinates;
        setSelected({ ...feature.properties, lng, lat });
      },
      updateTriggers: {
        getLineColor: [selected?.id],
        getLineWidth: [selected?.id],
      },
    });

    return [basemap, points];
  }, [features, selected, setSelected]);

  return (
    <section className="map-canvas">
      <DeckGL
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        layers={layers}
        getCursor={({ isHovering, isDragging }: { isHovering: boolean; isDragging: boolean }) =>
          isDragging ? 'grabbing' : isHovering ? 'pointer' : 'grab'
        }
      />

      {hover && (
        <div className="map-tooltip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
          <strong>{hover.feature.properties.street_name}</strong>
          <span>{hover.feature.properties.county} County</span>
          <span style={{ color: SEVERITY_COLORS[hover.feature.properties.severity_label] }}>
            {hover.feature.properties.severity_label} · index {hover.feature.properties.severity_index.toFixed(2)}
          </span>
          <span>
            TTC {hover.feature.properties.min_ttc.toFixed(1)}s · PET {hover.feature.properties.min_pet.toFixed(1)}s
          </span>
        </div>
      )}

      {loading && (
        <div className="map-loading">
          <CircularProgress size={28} />
        </div>
      )}

      <div className="map-attribution">
        © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> · © <a href="https://carto.com/attributions">CARTO</a>
      </div>
    </section>
  );
}
