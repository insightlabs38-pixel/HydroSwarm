import { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import cytoscape from 'cytoscape';
import type { NetworkRecord } from '../types';
import { fetchNetworks, importNetwork } from '../api/networks';
import { ApiError } from '../api/client';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { EmptyState } from '../components/common/EmptyState';
import { KeyValueGrid } from '../components/common/KeyValueGrid';

function coordinateCoverage(network: NetworkRecord): number | null {
  if (network.nodes.length === 0) return null;
  const withCoordinates = network.nodes.filter(
    (node) => node.coordinates[0] !== 0 || node.coordinates[1] !== 0,
  ).length;
  return withCoordinates / network.nodes.length;
}

function TopologyPreview({ network }: { network: NetworkRecord }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current || network.nodes.length === 0) return;
    const instance = cytoscape({
      container: ref.current,
      elements: [
        ...network.nodes.map((node) => ({ data: { id: node.nodeId } })),
        ...network.links
          .filter(
            (link) =>
              network.nodes.some((n) => n.nodeId === link.startNode) &&
              network.nodes.some((n) => n.nodeId === link.endNode),
          )
          .map((link) => ({
            data: { id: link.linkId, source: link.startNode, target: link.endNode },
          })),
      ],
      style: [
        {
          selector: 'node',
          style: {
            label: 'data(id)',
            'background-color': '#6bd6dd',
            color: '#d8f3f5',
            'font-size': 9,
            'text-valign': 'bottom',
            'text-margin-y': 5,
            width: 14,
            height: 14,
          },
        },
        {
          selector: 'edge',
          style: {
            width: 1.5,
            'line-color': '#456271',
            'target-arrow-color': '#6bd6dd',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
          },
        },
      ],
      layout: { name: 'breadthfirst', directed: true, spacingFactor: 1.1 },
    });
    return () => instance.destroy();
  }, [network]);

  if (network.nodes.length === 0) {
    return <EmptyState title="No topology metadata available for this network." />;
  }
  return (
    <div
      ref={ref}
      className="topology-canvas"
      role="img"
      aria-label={`Directed topology preview for network ${network.name}: ${network.nodeCount} nodes, ${network.linkCount} links`}
    />
  );
}

function NetworkDetail({ network }: { network: NetworkRecord }) {
  const coverage = coordinateCoverage(network);
  return (
    <>
      <div className="decision-badges">
        <StatusBadge tone={network.valid ? 'good' : 'danger'}>
          {network.valid ? 'VALID' : 'INVALID'}
        </StatusBadge>
      </div>
      <KeyValueGrid
        entries={[
          { key: 'nodes', label: 'Nodes', value: String(network.nodeCount) },
          { key: 'links', label: 'Links', value: String(network.linkCount) },
          { key: 'version', label: 'Version', value: String(network.version) },
          {
            key: 'coverage',
            label: 'Coordinate coverage',
            value: coverage === null ? 'not measured' : `${Math.round(coverage * 100)}%`,
          },
          { key: 'sha256', label: 'SHA-256', value: network.sha256, hash: true },
          { key: 'validated-at', label: 'Validated at', value: network.validatedAt },
        ]}
      />
      {network.validationErrors.length > 0 && (
        <ul className="warning-list">
          {network.validationErrors.map((error) => (
            <li key={error}>{error}</li>
          ))}
        </ul>
      )}
      <TopologyPreview network={network} />
    </>
  );
}

/**
 * ui-work.txt UI-9 / 17: network governance utility. Local file import
 * only ("no cloud upload"); a network must validate before incident use.
 * Every value here is either the real GET /api/networks list or the
 * real POST /api/networks/import response -- never fabricated.
 */
export function NetworkWorkspace() {
  const [selectedNetworkId, setSelectedNetworkId] = useState<string | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);

  const networksQuery = useQuery({
    queryKey: ['networks'],
    queryFn: ({ signal }) => fetchNetworks(signal),
  });
  const importMutation = useMutation({
    mutationFn: (file: File) => importNetwork(file),
    onSuccess: (record) => {
      setSelectedNetworkId(record.networkId);
      void networksQuery.refetch();
    },
  });

  const networks = networksQuery.data ?? [];
  const selected = networks.find((network) => network.networkId === selectedNetworkId) ?? null;

  return (
    <div className="workspace-grid">
      <Panel title="Import network" eyebrow="LOCAL FILE ONLY" className="wide-panel">
        <p className="supporting">
          Local .inp file only -- no cloud upload. A network must validate before it can be used for
          an incident.
        </p>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            const input = event.currentTarget.elements.namedItem('inp-file') as HTMLInputElement;
            const file = input.files?.[0];
            if (!file) {
              setFileError('Choose a .inp file first.');
              return;
            }
            setFileError(null);
            importMutation.mutate(file);
          }}
        >
          <input type="file" name="inp-file" accept=".inp" aria-label="EPANET .inp file" />
          <button type="submit" disabled={importMutation.isPending}>
            {importMutation.isPending ? 'Importing…' : 'Import'}
          </button>
        </form>
        {fileError && (
          <p className="supporting" role="alert">
            {fileError}
          </p>
        )}
        {importMutation.isError && (
          <p className="supporting" role="alert">
            {importMutation.error instanceof ApiError
              ? importMutation.error.message
              : 'Import failed.'}
          </p>
        )}
        {importMutation.isSuccess && (
          <p className="supporting" role="status">
            Imported {importMutation.data.name} ({importMutation.data.nodeCount} nodes,{' '}
            {importMutation.data.linkCount} links).
          </p>
        )}
      </Panel>
      <Panel
        title="Imported networks"
        eyebrow={`${networks.length} NETWORKS`}
        className="wide-panel"
      >
        {networksQuery.isLoading ? (
          <p role="status">Loading networks…</p>
        ) : networksQuery.isError ? (
          <EmptyState
            title="Networks unavailable."
            detail={(networksQuery.error as Error).message}
          />
        ) : networks.length === 0 ? (
          <EmptyState title="No networks imported yet." />
        ) : (
          <div className="table-scroll">
            <table className="benchmark-table">
              <caption>Locally imported EPANET networks</caption>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Nodes</th>
                  <th>Links</th>
                  <th>Version</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {networks.map((network) => (
                  <tr
                    key={network.networkId}
                    className={network.networkId === selectedNetworkId ? 'selected-row' : ''}
                  >
                    <th scope="row">
                      <button
                        type="button"
                        className="table-plan-button"
                        onClick={() => setSelectedNetworkId(network.networkId)}
                      >
                        {network.name}
                      </button>
                    </th>
                    <td>{network.nodeCount}</td>
                    <td>{network.linkCount}</td>
                    <td>{network.version}</td>
                    <td>
                      <StatusBadge tone={network.valid ? 'good' : 'danger'}>
                        {network.valid ? 'VALID' : 'INVALID'}
                      </StatusBadge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
      <Panel
        title={selected ? selected.name : 'Network detail'}
        eyebrow="VALIDATION"
        className="wide-panel"
      >
        {selected ? (
          <NetworkDetail network={selected} />
        ) : (
          <EmptyState title="Select a network above to see its validation detail." />
        )}
      </Panel>
    </div>
  );
}
