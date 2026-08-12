import { FilterProvider } from '../context/FilterContext';
import FilterBar from '../components/FilterBar/FilterBar';
import MapCanvas from '../components/MapCanvas/MapCanvas';
import DataGrid from '../components/DataGrid/DataGrid';
import SeverityDrawer from '../components/SeverityDrawer/SeverityDrawer';

export default function SSAMSafetyDashboard() {
  return (
    <FilterProvider>
      <div className="dashboard">
        <FilterBar />
        <div className="main-area">
          <MapCanvas />
          <SeverityDrawer />
        </div>
        <DataGrid />
      </div>
    </FilterProvider>
  );
}
