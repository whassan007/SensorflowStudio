export type SeverityLabel = 'Critical' | 'High' | 'Medium' | 'Low';

export interface SSAMRecord {
  id: string;
  street_name: string;
  county: string;
  lat: number;
  lng: number;
  conflict_type: string;
  min_ttc: number;
  min_pet: number;
  max_speed: number;
  severity_index: number;
  severity_label: SeverityLabel;
  manual_annotation?: string;
}

export interface StatewideQuery {
  counties?: string[];
  conflict_types?: string[];
  severity_labels?: string[];
  ttc_max?: number | null;
  speed_min?: number | null;
  page: number;
  page_size: number;
  sort_by: string;
  sort_dir: 'asc' | 'desc';
  search?: string;
}

export interface GeoJSONFeature {
  type: 'Feature';
  geometry: { type: 'Point'; coordinates: [number, number] };
  properties: Omit<SSAMRecord, 'lat' | 'lng' | 'manual_annotation'>;
}

export interface StatewideResponse {
  status: string;
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  rows: SSAMRecord[];
  geojson: { type: 'FeatureCollection'; features: GeoJSONFeature[] };
  summary: Record<SeverityLabel, number>;
  filter_options: {
    counties: string[];
    conflict_types: string[];
    severity_labels: SeverityLabel[];
  };
}

export const SEVERITY_COLORS: Record<SeverityLabel, string> = {
  Critical: '#ff1744',
  High: '#ff9100',
  Medium: '#ffea00',
  Low: '#00e676',
};

export const SEVERITY_RGB: Record<SeverityLabel, [number, number, number]> = {
  Critical: [255, 23, 68],
  High: [255, 145, 0],
  Medium: [255, 234, 0],
  Low: [0, 230, 118],
};
