// [SHARED] Do not remove or modify — this icon is the common sidebar icon
// for all community plugins. Changes here affect every plugin's navigation.
import React from 'react';

const CommunityIcon: React.FC = () => (
  <svg
    className="pf-v6-svg"
    viewBox="0 0 32 32"
    fill="currentColor"
    aria-hidden="true"
    role="img"
    width="1em"
    height="1em"
  >
    {/* Center person */}
    <circle cx="16" cy="7" r="4.2" />
    <path d="M9.5 21c0-3.6 2.9-6.5 6.5-6.5s6.5 2.9 6.5 6.5" />
    {/* Left person */}
    <circle cx="5.5" cy="11" r="3.2" />
    <path d="M1 23.5c0-2.5 2-4.5 4.5-4.5s4.5 2 4.5 4.5" />
    {/* Right person */}
    <circle cx="26.5" cy="11" r="3.2" />
    <path d="M22 23.5c0-2.5 2-4.5 4.5-4.5s4.5 2 4.5 4.5" />
  </svg>
);

export default CommunityIcon;
