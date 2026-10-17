"""Saved lifecycle projection and scientific plots, without mission execution."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import hashlib
import json
import numpy as np

from orbexa.utils.io_utils import convert_numpy_to_list


def _rotation(quaternion):
    from scipy.spatial.transform import Rotation
    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,) or not np.all(np.isfinite(value)) or abs(np.linalg.norm(value)-1) > 1e-8:
        raise ValueError('Recorded physical poses require unit xyzw body-to-ECI quaternions')
    return Rotation.from_quat(value).as_matrix()


def _contact_projection(attachment, approach, approach_digest, assembly, first_state):
    """Retain actual SI poses; validate the two literal coordinate handoffs."""
    contact = attachment.get('audit', {}).get('finite_contact')
    if contact is None:
        return None
    if contact.get('audit', {}).get('model') != 'finite_coupled_rigid_body_compliant_adhesive.v1':
        raise ValueError('Unsupported recorded finite-contact model')
    history = contact.get('history', [])
    if not history or not contact.get('success') or not attachment.get('success'):
        raise ValueError('An assembled replay requires its successful actual finite-contact history')
    times = np.asarray([row['time_s'] for row in history], dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('Finite-contact physical samples must have increasing finite times')
    ids = tuple(approach['actual_trajectories'])
    elements = {row['name']: row for row in assembly['elements']}
    if set(elements) != {'target', *ids}:
        raise ValueError('Finite-contact/assembled histories must name the exact source fleet')
    count = len(times)
    def states(name):
        values = [row['target'] if name == 'target' else row['chasers'][name] for row in history]
        result = {}
        for field, width in (('position_eci_m', 3), ('velocity_eci_m_s', 3),
                             ('quaternion_body_to_eci', 4), ('angular_velocity_body_rad_s', 3)):
            value = np.asarray([row[field] for row in values], dtype=float)
            if value.shape != (count, width) or not np.all(np.isfinite(value)):
                raise ValueError('Finite contact lacks aligned finite SI body poses: '+name)
            result[field] = value.tolist()
        for row in values:
            _rotation(row['quaternion_body_to_eci'])
        if any(abs(float(row['time_s'])-time) > 1e-9 for row, time in zip(values, times)):
            raise ValueError('Finite-contact body poses disagree with their physical sample clock')
        fuel = np.asarray([row['fuel_kg'] for row in values], dtype=float)
        if not np.all(np.isfinite(fuel)) or np.any(fuel < 0):
            raise ValueError('Finite-contact fuel samples must be finite and nonnegative')
        result['fuel_kg'] = fuel.tolist()
        return result
    for row in history:
        if set(row['chasers']) != set(ids) or set(row['pads']) != set(ids):
            raise ValueError('Every finite-contact sample must contain the exact source fleet and pad tips')
    target, bodies = states('target'), {key: states(key) for key in ids}
    kinematics = approach.get('metadata', {}).get('target_kinematics', {})
    saved_q, source_q = np.asarray(kinematics.get('anomaly', [])), approach['anom_history'][-1]
    rotations = np.asarray(kinematics.get('rotations', []), dtype=float)
    if (saved_q.ndim != 1 or not len(saved_q) or saved_q[-1] != source_q or rotations.shape != (len(saved_q), 3, 3)):
        raise ValueError('Finite-contact display frame requires the source approach recorded target rotation')
    relative_rotation = rotations[-1]
    if (not np.all(np.isfinite(relative_rotation)) or
            not np.allclose(relative_rotation.T@relative_rotation, np.eye(3), atol=1e-8, rtol=0) or
            abs(np.linalg.det(relative_rotation)-1) > 1e-8):
        raise ValueError('Source target-to-reference rotation must be proper')
    inertial_states = kinematics.get('target_inertial_states', kinematics.get('cartesian_state'))
    inertial_quaternions = kinematics.get('inertial_quaternions', kinematics.get('body_to_eci_quaternion'))
    if approach.get('metadata', {}).get('model_version') == 'nonlinear-reference-relative-v1' and (inertial_states is None or inertial_quaternions is None):
        raise ValueError('Nonlinear finite contact requires the actual retained inertial target boundary')
    if inertial_states is not None:
        inertial_states = np.asarray(inertial_states, dtype=float)
        if (inertial_states.shape != (len(saved_q), 6) or not np.all(np.isfinite(inertial_states)) or
                np.linalg.norm(inertial_states[-1, :3]-target['position_eci_m'][0]) > 5e-7 or
                np.linalg.norm(inertial_states[-1, 3:]-target['velocity_eci_m_s'][0]) > 1e-8):
            raise ValueError('Finite-contact target SI state differs from the saved actual source ephemeris')
    if inertial_quaternions is not None:
        if np.linalg.norm(_rotation(inertial_quaternions[-1])-_rotation(target['quaternion_body_to_eci'][0]), ord=2) > 1e-8:
            raise ValueError('Finite-contact target attitude differs from the actual source boundary')
    display_rotation = relative_rotation@_rotation(target['quaternion_body_to_eci'][0]).T
    source_times = approach.get('metadata', {}).get('elapsed_time_seconds', [])
    if not source_times or abs(float(source_times[-1])-times[0]) > 1e-8:
        raise ValueError('Finite contact must start at its actual source approach physical epoch')
    endpoint = {}
    for key in ids:
        point = np.asarray(approach['actual_trajectories'][key][-1][:3], dtype=float)
        physical = display_rotation@(np.asarray(bodies[key]['position_eci_m'][0])-target['position_eci_m'][0])
        if point.shape != (3,) or np.linalg.norm(point-physical) > 5e-7:
            raise ValueError('Finite-contact initial body positions differ from their actual source approach')
        endpoint[key] = point.tolist()
    if abs(times[-1]-float(first_state['time_s'])) > 1e-8:
        raise ValueError('Finite-contact final epoch must equal the first recorded rigid assembly epoch')
    assembly_rotation = _rotation(first_state['quaternion_body_to_eci'])
    center = np.asarray(first_state['center_of_mass_body_m'], dtype=float)
    origin = np.asarray(first_state['position_eci_m'], dtype=float)
    for name, body in dict(target=target, **bodies).items():
        element = elements[name]
        position = origin+assembly_rotation@(np.asarray(element['position_body_m'])-center)
        rotation = assembly_rotation@np.asarray(element['rotation_element_to_assembly'])
        if (np.linalg.norm(position-body['position_eci_m'][-1]) > 5e-7 or
                np.linalg.norm(rotation-_rotation(body['quaternion_body_to_eci'][-1]), ord=2) > 1e-8):
            raise ValueError('Rigid coordinate handoff changes an actual finite-contact body pose: '+name)
    pads = {}
    for key in ids:
        values = [row['pads'][key] for row in history]
        entry = {}
        for field, width in (('tip_position_eci_m', 3), ('extension_body_m', 3),
                             ('force_eci_n', 3), ('moment_eci_nm', 3)):
            value = np.asarray([row[field] for row in values], dtype=float)
            if value.shape != (count, width) or not np.all(np.isfinite(value)):
                raise ValueError('Finite contact requires aligned actual pad-tip/wrench records')
            entry[field] = value.tolist()
        for field in ('gap_m', 'activation'):
            value = np.asarray([row[field] for row in values], dtype=float)
            if value.shape != (count,) or not np.all(np.isfinite(value)):
                raise ValueError('Finite contact requires finite gap/activation histories')
            entry[field] = value.tolist()
        if np.any(np.asarray(entry['activation']) < 0) or np.any(np.asarray(entry['activation']) > 1):
            raise ValueError('Recorded finite-contact activation is outside its declared unit interval')
        initial_body, final_body = bodies[key], bodies[key]
        entry['mount_body_m'] = (_rotation(initial_body['quaternion_body_to_eci'][0]).T@
            (np.asarray(entry['tip_position_eci_m'][0])-initial_body['position_eci_m'][0])-entry['extension_body_m'][0]).tolist()
        entry['latched_tip_body_m'] = (_rotation(final_body['quaternion_body_to_eci'][-1]).T@
            (np.asarray(entry['tip_position_eci_m'][-1])-final_body['position_eci_m'][-1])).tolist()
        pads[key] = entry
    history_digest = hashlib.sha256(json.dumps(history, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return dict(schema='orbexa.finite-contact.studio.v1', time_s=times.tolist(), target=target, chasers=bodies, pads=pads,
        display_rotation_eci_to_capture=display_rotation.tolist(),
        approach_canonical_sha256=approach_digest, source_history_sha256=history_digest,
        approach_binding=dict(final_anomaly=source_q, physical_time_s=float(source_times[-1]),
            positions_reference_m=endpoint, target_rotation_reference=relative_rotation.tolist()),
        frame='Recorded ECI SI poses; display uses capture-fixed reference axes and the current target center as origin',
        interpolation='None; every displayed body and tip pose is a retained physical sample',
        latch_time_s=float(times[-1]), success=True)


class LifecycleDataset:
    """Physical assembly histories with explicit fixture/mission provenance."""

    def __init__(self, result):
        # Match persistence's recursive NumPy conversion before hashing or
        # projecting live records, including arrays nested in approach metadata.
        self.result = convert_numpy_to_list(
            result.to_dict() if hasattr(result, 'to_dict') else dict(result))

    def to_dict(self):
        r = self.result
        from orbexa.simulation.mission_result import ADTMPCMissionResult
        source_approach = ADTMPCMissionResult.from_dict(r['approach']).to_dict()
        approach_digest = hashlib.sha256(json.dumps(source_approach, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
        lifecycle_digest = hashlib.sha256(json.dumps(r, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
        history = r.get('assembly_history', [])
        if not history:
            raise ValueError('No recorded assembly states are available')
        times = np.array([x['time_s'] for x in history], dtype=float)
        positions = np.array([x['position_eci_m'] for x in history], dtype=float)
        velocities = np.array([x['velocity_eci_m_s'] for x in history], dtype=float)
        rotations = np.array([x['quaternion_body_to_eci'] for x in history], dtype=float)
        rates = np.array([x['angular_velocity_body_rad_s'] for x in history], dtype=float)
        masses = np.array([x['mass_kg'] for x in history], dtype=float)
        fuel = np.array([x['total_fuel_kg'] for x in history], dtype=float)
        centers = np.array([x['center_of_mass_body_m'] for x in history], dtype=float)
        if (positions.shape != (len(times), 3) or velocities.shape != positions.shape or
                rotations.shape != (len(times), 4) or rates.shape != positions.shape or
                centers.shape != positions.shape or len(times) < 1 or
                np.any(np.diff(times) <= 0) or not all(np.all(np.isfinite(x)) for x in
                    (times, positions, velocities, rotations, rates, masses, fuel, centers)) or
                np.any(masses <= 0) or np.any(fuel < 0) or
                not np.allclose(np.linalg.norm(rotations, axis=1), 1., atol=1e-8, rtol=0)):
            raise ValueError('Lifecycle histories require aligned finite physical samples and increasing time')
        assembly = r['attachment']['assembly']
        finite_contact = _contact_projection(r['attachment'], source_approach, approach_digest, assembly, history[0])
        if finite_contact is not None:
            finite_contact['lifecycle_canonical_sha256'] = lifecycle_digest
        mu, earth_radius = float(assembly['mu_m3_s2']), float(assembly['earth_radius_m'])
        radius = np.linalg.norm(positions, axis=1)
        if (not np.isfinite(mu) or mu <= 0 or not np.isfinite(earth_radius) or earth_radius <= 0
                or np.any(radius <= 0) or np.any(fuel > masses)):
            raise ValueError('Lifecycle orbital constants, radius and wet/fuel masses must be physical')
        momentum = np.cross(positions, velocities)
        eccentricity = np.linalg.norm(np.cross(velocities, momentum) / mu - positions / radius[:, None], axis=1)
        perigee = np.sum(momentum**2, axis=1) / mu / (1 + eccentricity) - earth_radius
        configuration = r.get('configuration', {}).get('deorbit', {}).get('options', {})
        provenance = r.get('approach', {}).get('metadata', {}).get('qualification_provenance', '')
        synthetic = 'SYNTHETIC' in provenance.upper()
        capture = r.get('configuration', {}).get('capture', {})
        attachment = r.get('attachment', {})
        surface_contacts = attachment.get('audit', {}).get('surface_contacts', {})
        if not surface_contacts and attachment.get('success'):
            surface_contacts = {name: dict(surface_point_body_m=gate['surface_anchor_body_m'])
                for name, gate in attachment.get('gates', {}).items()
                if isinstance(gate, Mapping) and gate.get('passed') and 'surface_anchor_body_m' in gate}
        if not isinstance(surface_contacts, Mapping):
            raise ValueError('Surface contact records must be keyed by chaser')
        for contact in surface_contacts.values():
            if not isinstance(contact, Mapping):
                raise ValueError('Each surface contact requires its recorded target-body point')
            point = np.asarray(contact.get('surface_point_body_m'), dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise ValueError('Surface contact records require a finite target-body point')
        default_radius = float(capture.get('default_chaser', {}).get('body_radius_m', .15))
        body_radii = {component['name']: float(capture.get('chasers', {}).get(component['name'], {}).get('body_radius_m', default_radius))
                      for component in assembly['elements'] if component['name'] != 'target'}
        if not np.isfinite(default_radius) or default_radius <= 0 or any(not np.isfinite(value) or value <= 0 for value in body_radii.values()):
            raise ValueError('Each displayed chaser requires a finite positive recorded body radius')
        controls = (r.get('detumbling') or {}).get('controls', []) + (r.get('deorbit') or {}).get('controls', [])
        applied = []
        control_index = 0
        for time in times:
            while control_index < len(controls) and controls[control_index]['time_s'] + controls[control_index]['duration_s'] < time - 1e-8:
                control_index += 1
            control = controls[control_index] if control_index < len(controls) else None
            applied.append(control['allocation']['force_body_n'] if control is not None and control['time_s'] <= time + 1e-8 else [0.,0.,0.])
        return dict(schema='orbexa.lifecycle.studio.v1', time_s=times.tolist(),
            elapsed_s=(times - times[0]).tolist(), positions_eci_m=positions.tolist(),
            quaternions=rotations.tolist(), angular_speed_rad_s=np.linalg.norm(rates, axis=1).tolist(),
            mass_kg=masses.tolist(), fuel_kg=fuel.tolist(), center_of_mass_body_m=centers.tolist(),
            phases=[x['phase'] for x in history], altitude_m=(radius-earth_radius).tolist(),
            applied_force_body_n=applied,
            perigee_altitude_m=perigee.tolist(), earth_radius_m=earth_radius,
            components=assembly['elements'], target=r.get('approach', {}).get('target', {}),
            body_radius_m=default_radius, body_radii_m=body_radii,
            surface_contacts=surface_contacts, attachment_success=bool(attachment.get('success')),
            finite_contact=finite_contact,
            terminal_handoff=r.get('terminal_handoff'),
            approach_canonical_sha256=approach_digest,
            lifecycle_canonical_sha256=lifecycle_digest,
            stages=dict(attachment=bool(attachment.get('success')),
                detumbling=None if r.get('detumbling') is None else bool(r['detumbling'].get('success')),
                deorbit=None if r.get('deorbit') is None else bool(r['deorbit'].get('success'))),
            force_alignment='Applied force over the recorded hold ending at this node; the first node shows its following hold',
            entry_interface_altitude_m=configuration.get('entry_interface_altitude_m', 120000.),
            target_perigee_altitude_m=configuration.get('target_perigee_altitude_m', 80000.),
            outcome=dict(success=bool(r.get('success')), message=r.get('message', '')),
            provenance=provenance or 'Recorded physical approach and modeled post-capture stages',
            synthetic=synthetic, scope=r.get('scope', ''),
            capture_assumptions=r.get('attachment', {}).get('assumptions', []))


class LifecycleRenderer:
    """Export source-derived plots and a compact browser replay data file."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.skipped_artifacts = {}

    def render(self, result, *, animate: bool = False, fps: int = 12):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        data = LifecycleDataset(result).to_dict()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir/'replay-data.json').write_text(json.dumps(data, allow_nan=False), encoding='utf-8')
        t = np.asarray(data['elapsed_s']) / 60
        self.skipped_artifacts = {}
        with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False}), plt.style.context('dark_background'):
            figure, axes = plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
            axes[0,0].semilogy(t,np.maximum(data['angular_speed_rad_s'],1e-14),color='#5bd4bd')
            axes[0,0].set(ylabel='Body angular speed (rad/s)',title='External-thruster detumbling and pointing')
            axes[0,1].plot(t,data['fuel_kg'],color='#f5b567')
            axes[0,1].set(ylabel='Remaining modeled propellant (kg)',title='Propellant accounting')
            axes[1,0].plot(t,np.asarray(data['altitude_m'])/1000,label='Actual altitude',color='#74b9ff')
            axes[1,0].plot(t,np.asarray(data['perigee_altitude_m'])/1000,label='Osculating perigee',color='#f5b567')
            axes[1,0].axhline(data['entry_interface_altitude_m']/1000,color='#aaa',ls=':',label='Entry interface')
            axes[1,0].axhline(data['target_perigee_altitude_m']/1000,color='#5bd4bd',ls=':',label='Perigee target')
            axes[1,0].set(ylabel='Altitude (km)',title='Orbital lowering and descending handoff')
            axes[1,0].legend(fontsize=8)
            p=np.asarray(data['positions_eci_m'])/1e6
            ax=axes[1,1]
            ax.add_patch(plt.Circle((0,0),data['earth_radius_m']/1e6,color='#123d68'))
            ax.plot(p[:,0],p[:,1],color='#74b9ff',lw=1.6)
            ax.scatter(p[0,0],p[0,1],color='#5bd4bd',label='Attachment',s=25)
            ax.scatter(p[-1,0],p[-1,1],color='#f5b567',label='Final recorded state',s=25)
            ax.set(xlabel='ECI X (1000 km)',ylabel='ECI Y (1000 km)',title='Recorded orbital projection',aspect='equal')
            ax.legend(fontsize=8)
            for axis in axes.flat:axis.grid(alpha=.13)
            for axis in (axes[0,0],axes[0,1],axes[1,0]):axis.set_xlabel('Time after attachment (min)')
            prefix='Synthetic physics fixture' if data['synthetic'] else 'Recorded ADTMPC approach → physical lifecycle'
            outcome='Entry interface handoff' if data['outcome']['success'] else 'Incomplete physical lifecycle'
            figure.suptitle('ORBEX-A | '+prefix+' | '+outcome,fontsize=15)
            figure.savefig(self.output_dir/'mission-lifecycle.png',dpi=180,facecolor='#080d14')
            plt.close(figure)
        artifacts = dict(data=self.output_dir/'replay-data.json',plot=self.output_dir/'mission-lifecycle.png')
        if animate:
            artifacts.update(self._animation(data, fps))
        return artifacts

    def _animation(self, data, fps):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
        if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
            raise ValueError('fps must be a positive integer')
        times = np.asarray(data['elapsed_s'])
        desired = np.linspace(times[0], times[-1], min(180,len(times)))
        indices = set(np.minimum(np.searchsorted(times, desired),len(times)-1))
        indices.update(i for i in range(1,len(times)) if data['phases'][i] != data['phases'][i-1])
        indices = sorted(indices | {0,len(times)-1})
        p = np.asarray(data['positions_eci_m'])/1e6
        t = times/60
        with plt.style.context('dark_background'):
            figure, axes = plt.subplots(1,2,figsize=(11.5,5),constrained_layout=True)
            left,right=axes
            left.add_patch(plt.Circle((0,0),data['earth_radius_m']/1e6,color='#153c63'))
            left.plot(p[:,0],p[:,1],color='#74b9ff',alpha=.4)
            marker,=left.plot([],[],'o',color='#f5b567',ms=6)
            left.set(xlabel='ECI X (1000 km)',ylabel='ECI Y (1000 km)',aspect='equal')
            right.plot(t,np.asarray(data['altitude_m'])/1000,color='#74b9ff',label='Altitude')
            right.plot(t,np.asarray(data['perigee_altitude_m'])/1000,color='#f5b567',label='Perigee')
            right.axhline(data['entry_interface_altitude_m']/1000,color='#5bd4bd',ls=':',label='Entry interface')
            cursor=right.axvline(0,color='white',alpha=.5)
            right.set(xlabel='Time after attachment (min)',ylabel='Altitude (km)')
            right.legend(fontsize=8)
            for axis in axes:axis.grid(alpha=.13)
            status=figure.suptitle('ORBEX-A')
            def draw(i):
                marker.set_data([p[i,0]],[p[i,1]])
                cursor.set_xdata([t[i],t[i]])
                prefix='Synthetic physics fixture' if data['synthetic'] else 'Recorded mission lifecycle'
                status.set_text(f"ORBEX-A | {prefix}\n{data['phases'][i].replace('_',' ')} | {t[i]:.1f} min | fuel {data['fuel_kg'][i]:.3f} kg")
                return marker,cursor,status
            animation=FuncAnimation(figure,draw,frames=indices,interval=1000/fps,blit=False)
            gif=self.output_dir/'mission-lifecycle.gif'
            animation.save(gif,writer=PillowWriter(fps=fps),dpi=105)
            artifacts={'gif':gif}
            try:
                import shutil
                import tempfile
                import matplotlib
                encoder=shutil.which('ffmpeg')
                if encoder is None:
                    import imageio_ffmpeg
                    encoder=imageio_ffmpeg.get_ffmpeg_exe()
                with tempfile.TemporaryDirectory(prefix='orbexa-video-',dir=self.output_dir) as temporary, matplotlib.rc_context({'animation.ffmpeg_path':encoder}):
                    encoded=Path(temporary)/'mission-lifecycle.mp4'
                    animation.save(encoded,writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-pix_fmt','yuv420p']),dpi=120)
                    mp4=self.output_dir/'mission-lifecycle.mp4'
                    encoded.replace(mp4)
                    artifacts['mp4']=mp4
            except Exception as error:
                self.skipped_artifacts['mission-lifecycle.mp4']=f'{type(error).__name__}: {error}'
            finally:
                plt.close(figure)
            (self.output_dir/'animation-provenance.json').write_text(json.dumps({
                'recorded_sample_indices':list(map(int,indices)),'fps':fps,
                'playback':'Even physical-time frame selection plus every phase transition; no interpolated state',
                'source_samples':len(times),'synthetic':data['synthetic'],
                'skipped_artifacts':dict(self.skipped_artifacts)},indent=2),encoding='utf-8')
            return artifacts
