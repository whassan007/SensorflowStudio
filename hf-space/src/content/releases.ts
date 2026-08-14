/**
 * Product version + release notes for About UI.
 * Source of truth: releases.json (also loaded by GET /api/about).
 */
import catalog from './releases.json';

export interface AboutLinks {
  github: string;
  hf_space: string;
}

export interface ReleaseNotes {
  version: string;
  date: string;
  title: string;
  highlights: string[];
}

export interface AboutCatalog {
  name: string;
  version: string;
  description: string;
  links: AboutLinks;
  releases: ReleaseNotes[];
}

export const ABOUT_CATALOG: AboutCatalog = catalog;
export const APP_NAME = ABOUT_CATALOG.name;
export const APP_VERSION = ABOUT_CATALOG.version;
export const RELEASES: ReleaseNotes[] = ABOUT_CATALOG.releases;
