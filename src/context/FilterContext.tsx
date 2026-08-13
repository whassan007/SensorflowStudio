import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { fetchStatewide } from '../services/api';
import type { SSAMRecord, StatewideResponse } from '../types';

export interface Filters {
  counties: string[];
  conflictTypes: string[];
  severityLabels: string[];
  ttcMax: number | null;
  speedMin: number | null;
  search: string;
}

const DEFAULT_FILTERS: Filters = {
  counties: [],
  conflictTypes: [],
  severityLabels: [],
  ttcMax: null,
  speedMin: null,
  search: '',
};

interface FilterContextValue {
  filters: Filters;
  setFilters: (patch: Partial<Filters>) => void;
  resetFilters: () => void;
  page: number;
  setPage: (page: number) => void;
  pageSize: number;
  setPageSize: (size: number) => void;
  sortBy: string;
  sortDir: 'asc' | 'desc';
  setSort: (column: string) => void;
  data: StatewideResponse | null;
  loading: boolean;
  error: string | null;
  selected: SSAMRecord | null;
  setSelected: (record: SSAMRecord | null) => void;
  refresh: () => void;
}

const FilterContext = createContext<FilterContextValue | null>(null);

export function FilterProvider({ children }: { children: React.ReactNode }) {
  const [filters, setFiltersState] = useState<Filters>(DEFAULT_FILTERS);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [sortBy, setSortBy] = useState('severity_index');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [data, setData] = useState<StatewideResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<SSAMRecord | null>(null);
  const [refreshTick, setRefreshTick] = useState(0);

  const setFilters = useCallback((patch: Partial<Filters>) => {
    setFiltersState((prev) => ({ ...prev, ...patch }));
    setPage(1);
  }, []);

  const resetFilters = useCallback(() => {
    setFiltersState(DEFAULT_FILTERS);
    setPage(1);
  }, []);

  const setSort = useCallback((column: string) => {
    setSortBy((prevBy) => {
      setSortDir((prevDir) => (prevBy === column && prevDir === 'desc' ? 'asc' : 'desc'));
      return column;
    });
    setPage(1);
  }, []);

  const refresh = useCallback(() => setRefreshTick((t) => t + 1), []);

  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    // Debounce so fast typing in the search box issues a single request.
    const timer = setTimeout(() => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setLoading(true);
      fetchStatewide(
        {
          counties: filters.counties.length ? filters.counties : undefined,
          conflict_types: filters.conflictTypes.length ? filters.conflictTypes : undefined,
          severity_labels: filters.severityLabels.length ? filters.severityLabels : undefined,
          ttc_max: filters.ttcMax,
          speed_min: filters.speedMin,
          search: filters.search || undefined,
          page,
          page_size: pageSize,
          sort_by: sortBy,
          sort_dir: sortDir,
        },
        controller.signal,
      )
        .then((res) => {
          setData(res);
          setError(null);
        })
        .catch((err: unknown) => {
          if ((err as Error).name !== 'AbortError') {
            setError(err instanceof Error ? err.message : String(err));
          }
        })
        .finally(() => {
          if (abortRef.current === controller) setLoading(false);
        });
    }, 250);
    return () => clearTimeout(timer);
  }, [filters, page, pageSize, sortBy, sortDir, refreshTick]);

  const value = useMemo<FilterContextValue>(
    () => ({
      filters,
      setFilters,
      resetFilters,
      page,
      setPage,
      pageSize,
      setPageSize,
      sortBy,
      sortDir,
      setSort,
      data,
      loading,
      error,
      selected,
      setSelected,
      refresh,
    }),
    [filters, setFilters, resetFilters, page, pageSize, sortBy, sortDir, setSort, data, loading, error, selected, refresh],
  );

  return <FilterContext.Provider value={value}>{children}</FilterContext.Provider>;
}

export function useFilters(): FilterContextValue {
  const ctx = useContext(FilterContext);
  if (!ctx) throw new Error('useFilters must be used within FilterProvider');
  return ctx;
}
